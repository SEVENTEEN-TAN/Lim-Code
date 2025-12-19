/**
 * LimCode - 对话处理器
 *
 * 负责处理对话请求，协调各个模块
 */

import { t } from '../../../i18n';
import type { ConfigManager } from '../../config/ConfigManager';
import type { ChannelManager } from '../../channel/ChannelManager';
import type { ConversationManager } from '../../conversation/ConversationManager';
import type { ToolRegistry } from '../../../tools/ToolRegistry';
import type { CheckpointManager, CheckpointRecord } from '../../checkpoint';
import type { SettingsManager } from '../../settings/SettingsManager';
import type { McpManager } from '../../mcp/McpManager';
import { PromptManager } from '../../prompt';
import { StreamAccumulator } from '../../channel/StreamAccumulator';
import type { Content, ContentPart } from '../../conversation/types';
import type { GetHistoryOptions } from '../../conversation/ConversationManager';
import type { BaseChannelConfig } from '../../config/configs/base';
import type { StreamChunk, GenerateResponse } from '../../channel/types';
import { ChannelError, ErrorType } from '../../channel/types';
import { parseXMLToolCalls } from '../../../tools/xmlFormatter';
import { parseJSONToolCalls, TOOL_CALL_START } from '../../../tools/jsonFormatter';
import { getDiffManager } from '../../../tools/file/diffManager';
import { getMultimodalCapability, type MultimodalCapability, type ChannelType as UtilChannelType, type ToolMode as UtilToolMode } from '../../../tools/utils';
import type {
    ChatRequestData,
    ChatSuccessData,
    ChatErrorData,
    ChatStreamChunkData,
    ChatStreamCompleteData,
    ChatStreamErrorData,
    ChatStreamToolIterationData,
    ChatStreamCheckpointsData,
    ChatStreamToolConfirmationData,
    ChatStreamToolsExecutingData,
    ToolConfirmationResponseData,
    PendingToolCall,
    RetryRequestData,
    EditAndRetryRequestData,
    DeleteToMessageRequestData,
    DeleteToMessageSuccessData,
    DeleteToMessageErrorData,
    AttachmentData,
    SummarizeContextRequestData,
    SummarizeContextSuccessData,
    SummarizeContextErrorData
} from './types';

/** 最大工具调用循环次数，防止无限循环 */
const MAX_TOOL_CALL_ITERATIONS = 20;

/**
 * 对话回合信息
 *
 * 一个回合从非函数响应的用户消息开始，到下一个非函数响应的用户消息之前结束
 */
interface ConversationRound {
    /** 回合起始消息索引 */
    startIndex: number;
    /** 回合结束消息索引（不包含） */
    endIndex: number;
    /** 该回合内助手消息的累计 token 数（取最后一个有 totalTokenCount 的助手消息） */
    tokenCount?: number;
}

/** 生成唯一的工具调用 ID */
function generateToolCallId(): string {
    return `fc_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * 对话处理器
 *
 * 职责：
 * 1. 接收对话请求
 * 2. 保存用户消息到历史
 * 3. 调用 AI API
 * 4. 处理工具调用（自动执行并返回结果）
 * 5. 保存 AI 响应到历史
 * 6. 返回响应
 */
export class ChatHandler {
    private checkpointManager?: CheckpointManager;
    private settingsManager?: SettingsManager;
    private mcpManager?: McpManager;
    private promptManager: PromptManager;
    
    constructor(
        private configManager: ConfigManager,
        private channelManager: ChannelManager,
        private conversationManager: ConversationManager,
        private toolRegistry?: ToolRegistry
    ) {
        this.promptManager = new PromptManager();
    }
    
    /**
     * 设置检查点管理器（可选）
     */
    setCheckpointManager(checkpointManager: CheckpointManager): void {
        this.checkpointManager = checkpointManager;
    }
    
    /**
     * 设置设置管理器（可选）
     */
    setSettingsManager(settingsManager: SettingsManager): void {
        this.settingsManager = settingsManager;
    }
    
    /**
     * 设置 MCP 管理器（可选）
     */
    setMcpManager(mcpManager: McpManager): void {
        this.mcpManager = mcpManager;
    }
    
    /**
     * 处理非流式对话请求
     * 支持工具调用循环：当 AI 返回工具调用时，自动执行工具并将结果返回给 AI
     *
     * @param request 对话请求数据
     * @returns 对话响应数据
     */
    async handleChat(request: ChatRequestData): Promise<ChatSuccessData | ChatErrorData> {
        try {
            const { conversationId, configId, message } = request;
            
            // 1. 确保对话存在（自动创建）
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
            }
            
            if (!config.enabled) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
            }
            
            // 3. 添加用户消息到历史（包含附件）
            const userParts = this.buildUserMessageParts(message, request.attachments);
            await this.conversationManager.addMessage(
                conversationId,
                'user',
                userParts
            );
            
            // 4. 工具调用循环
            const chatHistoryOptions = this.buildHistoryOptions(config);
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;
                
                // 获取对话历史
                const history = await this.conversationManager.getHistoryForAPI(conversationId, chatHistoryOptions);
                
                // 调用 AI（非流式）
                const response = await this.channelManager.generate({
                    configId,
                    history
                });
                
                // 类型守卫：确保是 GenerateResponse
                if (!('content' in response)) {
                    throw new Error('Unexpected stream response from generate()');
                }
                
                const generateResponse = response as GenerateResponse;
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, generateResponse.content);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(generateResponse.content);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，结束循环并返回
                    return {
                        success: true,
                        content: generateResponse.content
                    };
                }
                
                // 有工具调用，执行工具并添加结果
                // 获取当前消息索引（AI 响应刚刚添加到历史）
                const currentHistory = await this.conversationManager.getHistoryRef(conversationId);
                const messageIndex = currentHistory.length - 1;
                
                const functionResponses = await this.executeFunctionCalls(functionCalls, conversationId, messageIndex);
                
                // 将函数响应添加到历史（作为 user 消息）
                await this.conversationManager.addMessage(
                    conversationId,
                    'user',
                    functionResponses
                );
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            return {
                success: false,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 错误处理
            return {
                success: false,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 格式化错误信息
     * 如果有详细错误信息（如 API 返回的响应体），直接追加显示
     */
    private formatError(error: unknown): { code: string; message: string } {
        if (error instanceof ChannelError) {
            let message = error.message;
            
            // 如果有详细错误信息，直接 JSON 序列化追加
            if (error.details) {
                try {
                    const detailsStr = typeof error.details === 'string'
                        ? error.details
                        : JSON.stringify(error.details, null, 2);
                    message = `${error.message}\n${detailsStr}`;
                } catch {
                    // 忽略序列化错误
                }
            }
            
            return {
                code: error.type || 'CHANNEL_ERROR',
                message
            };
        }
        
        const err = error as any;
        return {
            code: err.code || 'UNKNOWN_ERROR',
            message: err.message || t('modules.api.chat.errors.unknownError')
        };
    }
    
    /**
     * 处理流式对话请求（自动根据配置决定使用流式或非流式）
     * 支持工具调用循环：当 AI 返回工具调用时，自动执行工具并将结果返回给 AI
     *
     * @param request 对话请求数据
     * @returns 异步生成器，产生流式响应
     */
    async *handleChatStream(
        request: ChatRequestData
    ): AsyncGenerator<
        ChatStreamChunkData | ChatStreamCompleteData | ChatStreamErrorData | ChatStreamToolIterationData | ChatStreamCheckpointsData | ChatStreamToolConfirmationData | ChatStreamToolsExecutingData
    > {
        try {
            const { conversationId, configId, message } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
                return;
            }
            
            if (!config.enabled) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
                return;
            }
            
            // 3. 中断之前未完成的 diff 等待
            const diffManager = getDiffManager();
            diffManager.markUserInterrupt();
            
            // 4. 为用户消息创建存档点（如果配置了执行前）
            if (this.checkpointManager && this.settingsManager?.shouldCreateBeforeUserMessageCheckpoint()) {
                // 获取当前历史长度作为消息索引
                const currentHistoryBefore = await this.conversationManager.getHistoryRef(conversationId);
                const userMessageIndex = currentHistoryBefore.length;
                
                const checkpoint = await this.checkpointManager.createCheckpoint(
                    conversationId,
                    userMessageIndex,
                    'user_message',
                    'before'
                );
                if (checkpoint) {
                    // 立即发送用户消息前存档点到前端
                    yield {
                        conversationId,
                        checkpoints: [checkpoint],
                        checkpointOnly: true as const
                    };
                }
            }
            
            // 5. 添加用户消息到历史（包含附件）
            const userParts = this.buildUserMessageParts(message, request.attachments);
            await this.conversationManager.addMessage(
                conversationId,
                'user',
                userParts
            );
            
            // 6. 为用户消息创建存档点（如果配置了执行后）
            if (this.checkpointManager && this.settingsManager?.shouldCreateAfterUserMessageCheckpoint()) {
                const currentHistoryAfterUser = await this.conversationManager.getHistoryRef(conversationId);
                const userMessageIndexAfter = currentHistoryAfterUser.length - 1;
                
                const checkpoint = await this.checkpointManager.createCheckpoint(
                    conversationId,
                    userMessageIndexAfter,
                    'user_message',
                    'after'
                );
                if (checkpoint) {
                    // 立即发送用户消息后存档点到前端
                    yield {
                        conversationId,
                        checkpoints: [checkpoint],
                        checkpointOnly: true as const
                    };
                }
            }
            
            // 7. 重置中断标记并开始新的对话
            diffManager.resetUserInterrupt();
            
            // 8. 判断是否是首条消息（需要刷新动态系统提示词）
            const currentHistoryCheck = await this.conversationManager.getHistoryRef(conversationId);
            const isFirstMessage = currentHistoryCheck.length === 1;  // 只有刚添加的用户消息
            
            // 获取动态系统提示词（首条消息时刷新）
            const dynamicSystemPrompt = isFirstMessage
                ? this.promptManager.refreshAndGetPrompt()
                : this.promptManager.getSystemPrompt();
            
            // 9. 工具调用循环
            let iteration = 0;
            
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;

                // 检查是否已取消
                if (request.abortSignal?.aborted) {
                    // 发送 cancelled 消息给前端，让前端正确清理状态
                    yield {
                        conversationId,
                        cancelled: true as const
                    } as any;
                    return;
                }

                // 为模型消息创建存档点（如果配置了执行前）
                // 根据 modelOuterLayerOnly 设置决定是否在每次迭代都创建
                const outerLayerOnly = this.settingsManager?.isModelOuterLayerOnly() ?? true;
                const shouldCreateBeforeCheckpoint = this.checkpointManager &&
                    this.settingsManager?.shouldCreateBeforeModelMessageCheckpoint() &&
                    (!outerLayerOnly || iteration === 1);  // 只在最外层模式下，只在第一次迭代创建
                
                if (shouldCreateBeforeCheckpoint) {
                    const currentHistoryBeforeModel = await this.conversationManager.getHistoryRef(conversationId);
                    // 模型消息将被添加到 currentHistoryBeforeModel.length 位置
                    const modelMessageIndexBefore = currentHistoryBeforeModel.length;
                    
                    const checkpoint = await this.checkpointManager!.createCheckpoint(
                        conversationId,
                        modelMessageIndexBefore,
                        'model_message',
                        'before'
                    );
                    if (checkpoint) {
                        // 立即发送模型消息前存档点到前端
                        yield {
                            conversationId,
                            checkpoints: [checkpoint],
                            checkpointOnly: true as const
                        };
                    }
                }
                
                // 获取对话历史（应用上下文裁剪）
                const historyOptions = this.buildHistoryOptions(config);
                const history = await this.getHistoryWithContextTrim(conversationId, config, historyOptions);
                
                // 记录请求开始时间（用于计算响应持续时间）
                const requestStartTime = Date.now();
                
                // 调用 AI（传递 abortSignal 和动态系统提示词）
                const response = await this.channelManager.generate({
                    configId,
                    history,
                    abortSignal: request.abortSignal,
                    dynamicSystemPrompt
                });
                
                // 处理响应
                let finalContent: Content;
                
                if (this.isAsyncGenerator(response)) {
                    // 流式响应，累加器会自动从全局设置获取 toolMode
                    const accumulator = new StreamAccumulator();
                    // 设置请求开始时间，用于计算 responseDuration
                    accumulator.setRequestStartTime(requestStartTime);
                    // 根据配置类型设置 providerType（用于多格式思考签名存储）
                    accumulator.setProviderType(config.type as 'gemini' | 'openai' | 'anthropic' | 'custom');
                    let cancelled = false;
                    let lastPartsLength = 0;
                    
                    for await (const chunk of response) {
                        // 检查是否已取消
                        if (request.abortSignal?.aborted) {
                            cancelled = true;
                            break;
                        }
                        accumulator.add(chunk);
                        
                        // 获取累加器处理后的 parts（实时转换工具调用标记）
                        const currentContent = accumulator.getContent();
                        const currentParts = currentContent.parts;
                        
                        // 计算增量 delta
                        const newDelta = currentParts.slice(lastPartsLength);
                        lastPartsLength = currentParts.length;
                        
                        // 发送处理后的 chunk，附加 thinkingStartTime 供前端实时显示
                        const processedChunk: typeof chunk & { thinkingStartTime?: number } = {
                            ...chunk,
                            delta: newDelta.length > 0 ? newDelta : chunk.delta
                        };
                        
                        // 如果有思考开始时间，添加到 chunk
                        const thinkingStartTime = currentContent.thinkingStartTime;
                        if (thinkingStartTime !== undefined) {
                            processedChunk.thinkingStartTime = thinkingStartTime;
                        }
                        
                        yield { conversationId, chunk: processedChunk };
                        if (chunk.done) {
                            break;
                        }
                    }
                    
                    // 如果已取消，保存已接收的内容并 yield cancelled 消息
                    if (cancelled) {
                        const partialContent = accumulator.getContent();
                        if (partialContent.parts.length > 0) {
                            await this.conversationManager.addContent(conversationId, partialContent);
                            // yield cancelled 消息，包含 partialContent 以便前端更新计时信息
                            yield {
                                conversationId,
                                cancelled: true as const,
                                content: partialContent
                            } as any;
                        }
                        return;
                    }
                    
                    finalContent = accumulator.getContent();
                } else {
                    // 非流式响应
                    finalContent = (response as GenerateResponse).content;
                    // 添加响应持续时间
                    finalContent.responseDuration = Date.now() - requestStartTime;
                    finalContent.chunkCount = 1;
                    // 对于非流式，模拟一个完成块
                    yield {
                        conversationId,
                        chunk: {
                            delta: finalContent.parts,
                            done: true
                        }
                    };
                }
                
                // 转换 XML 工具调用为 functionCall 格式（如果有）
                this.convertXMLToolCallsToFunctionCalls(finalContent);
                
                // 为没有 id 的 functionCall 添加唯一 id（Gemini 格式不返回 id）
                this.ensureFunctionCallIds(finalContent);
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, finalContent);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(finalContent);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，为模型消息创建存档点（如果配置了执行后）
                    const modelMessageCheckpoints: CheckpointRecord[] = [];
                    if (this.checkpointManager && this.settingsManager?.shouldCreateAfterModelMessageCheckpoint()) {
                        const modelHistory = await this.conversationManager.getHistoryRef(conversationId);
                        const modelMessageIndex = modelHistory.length - 1;
                        
                        const checkpoint = await this.checkpointManager.createCheckpoint(
                            conversationId,
                            modelMessageIndex,
                            'model_message',
                            'after'
                        );
                        if (checkpoint) {
                            modelMessageCheckpoints.push(checkpoint);
                        }
                    }
                    
                    // 结束循环，返回完成数据（模型消息后的检查点）
                    yield {
                        conversationId,
                        content: finalContent,
                        checkpoints: modelMessageCheckpoints
                    };
                    return;
                }
                
                // 有工具调用，检查是否需要确认
                const toolsNeedingConfirmation = this.getToolsNeedingConfirmation(functionCalls);
                
                if (toolsNeedingConfirmation.length > 0) {
                    // 有工具需要确认，发送确认请求到前端
                    const pendingToolCalls: PendingToolCall[] = toolsNeedingConfirmation.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }));
                    
                    yield {
                        conversationId,
                        pendingToolCalls,
                        content: finalContent,
                        awaitingConfirmation: true as const
                    };
                    
                    // 暂停执行，等待前端调用 handleToolConfirmation
                    return;
                }
                
                // 不需要确认，直接执行工具
                // 获取当前消息索引（AI 响应刚刚添加到历史）
                const currentHistory = await this.conversationManager.getHistoryRef(conversationId);
                const messageIndex = currentHistory.length - 1;
                
                // 工具执行前先发送计时信息（让前端立即显示）
                yield {
                    conversationId,
                    content: finalContent,
                    toolsExecuting: true as const,
                    pendingToolCalls: functionCalls.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }))
                };
                
                const { responseParts, toolResults, checkpoints, multimodalAttachments } = await this.executeFunctionCallsWithResults(
                    functionCalls,
                    conversationId,
                    messageIndex,
                    config,
                    request.abortSignal
                );
                
                // 先将函数响应添加到历史（确保取消时也能保存）
                // 对于 XML/JSON 模式，如果有多模态附件，将其放在 parts 前面
                const functionResponseParts = multimodalAttachments
                    ? [...multimodalAttachments, ...responseParts]
                    : responseParts;
                    
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: functionResponseParts,
                    isFunctionResponse: true
                });
                
                // 检查是否有工具被取消
                const hasCancelled = toolResults.some(r => (r.result as any).cancelled);
                if (hasCancelled) {
                    // 有工具被取消，发送最终的 toolIteration 后结束
                    yield {
                        conversationId,
                        content: finalContent,
                        toolIteration: true as const,
                        toolResults,
                        checkpoints
                    };
                    return;
                }
                
                // 发送 toolIteration 信号（包含工具执行结果和检查点）
                yield {
                    conversationId,
                    content: finalContent,
                    toolIteration: true as const,
                    toolResults,
                    checkpoints
                };
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            yield {
                conversationId,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 检查是否是用户取消错误
            if (error instanceof ChannelError && error.type === ErrorType.CANCELLED_ERROR) {
                // 用户取消，不需要 yield error，直接返回
                // cancelled 消息已经在流式处理循环中 yield 过了
                return;
            }
            
            yield {
                conversationId: request.conversationId,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 将附件转换为 ContentPart[] 格式（Gemini inlineData）
     *
     * 存储时包含以下字段：
     * - id: 附件唯一标识，仅用于存储和前端显示
     * - name: 附件名称，仅用于存储和前端显示
     *
     * 注意：displayName 不在此处添加，因为它只适合用于 Gemini function call 格式。
     * 用户输入的附件在发送给 API 时，由 ConversationManager.getHistoryForAPI 统一处理。
     *
     * @param message 用户消息文本
     * @param attachments 附件列表
     * @returns ContentPart[] 包含文本和附件的 parts
     */
    private buildUserMessageParts(message: string, attachments?: AttachmentData[]): ContentPart[] {
        const parts: ContentPart[] = [];
        
        // 先添加附件（作为 inlineData，包含 id 和 name 用于存储和前端显示）
        // 注意：不添加 displayName，因为它只适用于 Gemini function call 格式
        if (attachments && attachments.length > 0) {
            for (const attachment of attachments) {
                parts.push({
                    inlineData: {
                        mimeType: attachment.mimeType,
                        data: attachment.data,
                        id: attachment.id,
                        name: attachment.name
                    }
                });
            }
        }
        
        // 再添加文本消息
        if (message) {
            parts.push({ text: message });
        }
        
        return parts;
    }
    
    /**
     * 确保 Content 中的所有 functionCall 都有唯一 id
     *
     * Gemini 格式的响应不包含 functionCall.id，
     * 我们在保存到历史之前为其添加唯一 id，
     * 以便前端可以通过 id 匹配 functionCall 和 functionResponse
     *
     * 对于 OpenAI 格式，如果已经有 id 则保留原有 id
     */
    private ensureFunctionCallIds(content: Content): void {
        for (const part of content.parts) {
            if (part.functionCall && !part.functionCall.id) {
                part.functionCall.id = generateToolCallId();
            }
        }
    }
    
    /**
     * 从 Content 中提取函数调用
     *
     * 支持两种模式：
     * 1. Function Call 模式：从 part.functionCall 提取
     * 2. XML 模式：从 part.text 中解析 <tool_use> XML 标签
     */
    private extractFunctionCalls(content: Content): Array<{
        name: string;
        args: Record<string, unknown>;
        id: string;
    }> {
        const calls: Array<{
            name: string;
            args: Record<string, unknown>;
            id: string;
        }> = [];
        
        for (const part of content.parts) {
            // Function Call 模式
            if (part.functionCall) {
                calls.push({
                    name: part.functionCall.name,
                    args: part.functionCall.args,
                    id: part.functionCall.id || generateToolCallId()
                });
            }
            
            // XML 模式：从文本中解析工具调用（支持多个调用）
            if (part.text && part.text.includes('<tool_use>')) {
                const xmlCalls = parseXMLToolCalls(part.text);
                for (const xmlCall of xmlCalls) {
                    calls.push({
                        name: xmlCall.name,
                        args: xmlCall.args,
                        id: generateToolCallId()
                    });
                }
            }
            
            // JSON 边界标记模式：从文本中解析工具调用（支持多个调用）
            if (part.text && part.text.includes(TOOL_CALL_START)) {
                const jsonCalls = parseJSONToolCalls(part.text);
                for (const jsonCall of jsonCalls) {
                    calls.push({
                        name: jsonCall.tool,
                        args: jsonCall.parameters,
                        id: generateToolCallId()
                    });
                }
            }
        }
        
        return calls;
    }
    
    /**
     * 将 XML/JSON 工具调用转换为 functionCall 格式
     *
     * 当使用 XML 或 JSON 模式时，模型返回的文本中包含工具调用标记。
     * 此方法将文本中的工具调用解析出来，并转换为 functionCall 格式的 parts。
     * 这样存储和前端显示都使用统一的 functionCall 格式。
     */
    private convertXMLToolCallsToFunctionCalls(content: Content): void {
        const newParts: ContentPart[] = [];
        
        for (const part of content.parts) {
            // 检查文本中是否包含 XML 工具调用（支持多个调用）
            if (part.text && part.text.includes('<tool_use>')) {
                const xmlCalls = parseXMLToolCalls(part.text);
                if (xmlCalls.length > 0) {
                    // 提取所有工具调用，并分离前后文本
                    let remainingText = part.text;
                    
                    for (const xmlCall of xmlCalls) {
                        // 查找当前工具调用的位置
                        const startMarkerIdx = remainingText.indexOf('<tool_use>');
                        const endMarkerIdx = remainingText.indexOf('</tool_use>');
                        
                        if (startMarkerIdx !== -1 && endMarkerIdx !== -1) {
                            // 添加工具调用前的文本
                            const textBefore = remainingText.substring(0, startMarkerIdx).trim();
                            if (textBefore) {
                                newParts.push({ text: textBefore });
                            }
                            
                            // 添加 functionCall
                            newParts.push({
                                functionCall: {
                                    name: xmlCall.name,
                                    args: xmlCall.args,
                                    id: generateToolCallId()
                                }
                            });
                            
                            // 更新剩余文本
                            remainingText = remainingText.substring(endMarkerIdx + '</tool_use>'.length);
                        }
                    }
                    
                    // 添加最后剩余的文本
                    const finalText = remainingText.trim();
                    if (finalText) {
                        newParts.push({ text: finalText });
                    }
                } else {
                    // 解析失败，保留原文本
                    newParts.push(part);
                }
            }
            // 检查文本中是否包含 JSON 边界标记工具调用
            else if (part.text && part.text.includes(TOOL_CALL_START)) {
                const jsonCalls = parseJSONToolCalls(part.text);
                if (jsonCalls.length > 0) {
                    // 提取所有工具调用，并分离前后文本
                    let remainingText = part.text;
                    
                    for (const jsonCall of jsonCalls) {
                        // 查找当前工具调用的位置
                        const startMarkerIdx = remainingText.indexOf(TOOL_CALL_START);
                        const endMarkerIdx = remainingText.indexOf('<<<END_TOOL_CALL>>>');
                        
                        if (startMarkerIdx !== -1 && endMarkerIdx !== -1) {
                            // 添加工具调用前的文本
                            const textBefore = remainingText.substring(0, startMarkerIdx).trim();
                            if (textBefore) {
                                newParts.push({ text: textBefore });
                            }
                            
                            // 添加 functionCall
                            newParts.push({
                                functionCall: {
                                    name: jsonCall.tool,
                                    args: jsonCall.parameters,
                                    id: generateToolCallId()
                                }
                            });
                            
                            // 更新剩余文本
                            remainingText = remainingText.substring(endMarkerIdx + '<<<END_TOOL_CALL>>>'.length);
                        }
                    }
                    
                    // 添加最后剩余的文本
                    const finalText = remainingText.trim();
                    if (finalText) {
                        newParts.push({ text: finalText });
                    }
                } else {
                    // 解析失败，保留原文本
                    newParts.push(part);
                }
            } else {
                // 不包含工具调用，保留原 part
                newParts.push(part);
            }
        }
        
        // 替换 parts
        content.parts = newParts;
    }
    
    /**
     * 执行函数调用并返回函数响应 parts
     *
     * @param calls 函数调用列表
     * @param conversationId 对话 ID（用于创建检查点）
     * @param messageIndex 消息索引（用于创建检查点）
     */
    private async executeFunctionCalls(
        calls: Array<{
            name: string;
            args: Record<string, unknown>;
            id: string;
        }>,
        conversationId?: string,
        messageIndex?: number
    ): Promise<ContentPart[]> {
        const { responseParts } = await this.executeFunctionCallsWithResults(calls, conversationId, messageIndex);
        return responseParts;
    }
    
    /**
     * 执行函数调用并返回函数响应 parts 和工具执行结果
     *
     * 检查点策略：
     * - 在所有工具执行前创建一个检查点（使用 'tool_batch' 作为 toolName）
     * - 在所有工具执行后创建一个检查点
     * - 这样一条消息无论有多少个工具调用，只会创建一对检查点
     *
     * 多模态数据处理：
     * - 对于 function_call 模式：使用 functionResponse.parts 包含多模态数据
     * - 对于 xml/json 模式：将多模态数据作为用户消息的 inlineData 附件发送
     *
     * @param calls 函数调用列表
     * @param conversationId 对话 ID（用于创建检查点）
     * @param messageIndex 消息索引（用于创建检查点）
     * @param config 渠道配置（用于获取多模态工具设置和工具模式）
     * @param abortSignal 取消信号（用于中断工具执行）
     */
    private async executeFunctionCallsWithResults(
        calls: Array<{
            name: string;
            args: Record<string, unknown>;
            id: string;
        }>,
        conversationId?: string,
        messageIndex?: number,
        config?: BaseChannelConfig,
        abortSignal?: AbortSignal
    ): Promise<{
        responseParts: ContentPart[];
        toolResults: Array<{
            id: string;
            name: string;
            result: Record<string, unknown>;
        }>;
        checkpoints: CheckpointRecord[];
        /** 多模态附件（仅 xml/json 模式时使用） */
        multimodalAttachments?: ContentPart[];
    }> {
        const responseParts: ContentPart[] = [];
        const toolResults: Array<{
            id: string;
            name: string;
            result: Record<string, unknown>;
        }> = [];
        const checkpoints: CheckpointRecord[] = [];
        const multimodalAttachments: ContentPart[] = [];
        
        // 获取工具调用模式
        const toolMode = config?.toolMode || 'function_call';
        const isPromptMode = toolMode === 'xml' || toolMode === 'json';
        
        // 确定检查点的工具名称
        // 如果只有一个工具调用，使用该工具名称
        // 如果有多个工具调用，使用 'tool_batch'
        const toolNameForCheckpoint = calls.length === 1 ? calls[0].name : 'tool_batch';
        
        // 在所有工具执行前创建一个检查点
        if (this.checkpointManager && conversationId !== undefined && messageIndex !== undefined) {
            const beforeCheckpoint = await this.checkpointManager.createCheckpoint(
                conversationId,
                messageIndex,
                toolNameForCheckpoint,
                'before'
            );
            if (beforeCheckpoint) {
                checkpoints.push(beforeCheckpoint);
            }
        }
        
        // 执行所有工具
        for (const call of calls) {
            // 检查是否已取消
            if (abortSignal?.aborted) {
                break;
            }
            
            let response: Record<string, unknown>;
            
            try {
                // 检查是否是 MCP 工具（格式：mcp__{serverId}__{toolName}）
                if (call.name.startsWith('mcp__') && this.mcpManager) {
                    const parts = call.name.split('__');
                    if (parts.length >= 3) {
                        const serverId = parts[1];
                        const toolName = parts.slice(2).join('__');
                        
                        const result = await this.mcpManager.callTool({
                            serverId,
                            toolName,
                            arguments: call.args
                        });
                        
                        if (result.success) {
                            // 将 MCP 响应转换为标准格式
                            const textContent = result.content
                                ?.filter(c => c.type === 'text')
                                .map(c => c.text)
                                .join('\n') || '';
                            
                            response = {
                                success: true,
                                content: textContent || t('modules.api.chat.errors.toolExecutionSuccess')
                            };
                        } else {
                            response = {
                                success: false,
                                error: result.error || t('modules.api.chat.errors.mcpToolCallFailed')
                            };
                        }
                    } else {
                        response = {
                            success: false,
                            error: t('modules.api.chat.errors.invalidMcpToolName', { toolName: call.name })
                        };
                    }
                } else {
                    // 查找内置工具
                    const tool = this.toolRegistry?.getTool(call.name);
                    
                    if (!tool) {
                        response = {
                            success: false,
                            error: t('modules.api.chat.errors.toolNotFound', { toolName: call.name })
                        };
                    } else {
                        // 获取渠道多模态能力
                        const channelType = (config?.type || 'custom') as UtilChannelType;
                        const currentToolMode = (toolMode || 'function_call') as UtilToolMode;
                        const multimodalEnabled = config?.multimodalToolsEnabled ?? false;
                        const capability = getMultimodalCapability(channelType, currentToolMode, multimodalEnabled);
                        
                        // 构建工具执行上下文，包含多模态配置、能力、取消信号和工具调用 ID
                        const toolContext: Record<string, unknown> = {
                            multimodalEnabled,
                            capability,
                            abortSignal,
                            toolId: call.id,  // 使用函数调用 ID 作为工具 ID，用于追踪和取消
                            toolOptions: config?.toolOptions  // 传递工具配置
                        };
                        
                        // 为特定工具添加配置
                        if (call.name === 'generate_image' && this.settingsManager) {
                            const imageConfig = this.settingsManager.getGenerateImageConfig();
                            // 添加代理配置
                            toolContext.config = {
                                ...imageConfig,
                                proxyUrl: this.settingsManager.getEffectiveProxyUrl()
                            };
                        }
                        
                        // remove_background 工具复用 generate_image 的 API 配置，但使用自己的返回图片配置
                        if (call.name === 'remove_background' && this.settingsManager) {
                            const imageConfig = this.settingsManager.getGenerateImageConfig();
                            const removeConfig = this.settingsManager.getRemoveBackgroundConfig();
                            // 添加代理配置
                            toolContext.config = {
                                ...imageConfig,
                                ...removeConfig,
                                proxyUrl: this.settingsManager.getEffectiveProxyUrl()
                            };
                        }
                        
                        // crop_image 工具配置
                        if (call.name === 'crop_image' && this.settingsManager) {
                            const cropConfig = this.settingsManager.getCropImageConfig();
                            toolContext.config = {
                                ...cropConfig
                            };
                        }
                        
                        // resize_image 工具配置
                        if (call.name === 'resize_image' && this.settingsManager) {
                            const resizeConfig = this.settingsManager.getResizeImageConfig();
                            toolContext.config = {
                                ...resizeConfig
                            };
                        }
                        
                        // rotate_image 工具配置
                        if (call.name === 'rotate_image' && this.settingsManager) {
                            const rotateConfig = this.settingsManager.getRotateImageConfig();
                            toolContext.config = {
                                ...rotateConfig
                            };
                        }
                        
                        // 执行工具
                        const result = await tool.handler(call.args, toolContext);
                        response = result as unknown as Record<string, unknown>;
                    }
                }
            } catch (error) {
                const err = error as Error;
                response = {
                    success: false,
                    error: err.message || t('modules.api.chat.errors.toolExecutionFailed')
                };
            }
            
            // 添加到工具结果（使用深拷贝，保留完整数据供前端显示）
            // 注意：后续会删除 response.multimodal，但 toolResults 需要保留原始数据
            toolResults.push({
                id: call.id,
                name: call.name,
                result: JSON.parse(JSON.stringify(response))
            });
            
            // 处理多模态数据
            const multimodalData = (response as any).multimodal as Array<{
                mimeType: string;
                data: string;
                name?: string;
            }> | undefined;
            
            // 根据工具模式和渠道类型处理多模态数据
            // 重要：始终将多模态数据存储到历史中，以便当前轮次可以发送给 AI
            // supportsHistoryMultimodal 只控制后续轮次是否发送历史中的多模态数据
            // 这个过滤在 getHistoryForAPI 中进行，而不是在存储时
            if (multimodalData && multimodalData.length > 0) {
                // 获取渠道能力
                const channelType = (config?.type || 'custom') as UtilChannelType;
                const currentToolMode = (toolMode || 'function_call') as UtilToolMode;
                const multimodalEnabled = config?.multimodalToolsEnabled ?? false;
                const capability = getMultimodalCapability(channelType, currentToolMode, multimodalEnabled);
                
                if (isPromptMode) {
                    // XML/JSON 模式：将多模态数据作为用户消息附件
                    // 始终存储多模态数据，以便当前轮次可以发送给 AI
                    // 后续轮次发送时，getHistoryForAPI 会根据 supportsHistoryMultimodal 决定是否过滤
                    // 使用 displayName 而不是 name（Gemini API 支持 displayName）
                    for (const item of multimodalData) {
                        multimodalAttachments.push({
                            inlineData: {
                                mimeType: item.mimeType,
                                data: item.data,
                                displayName: item.name
                            }
                        });
                    }
                    // 从响应中移除 multimodal 数据（因为已经单独处理）
                    delete (response as any).multimodal;
                } else {
                    // function_call 模式
                    // 检查渠道是否支持 function_call 模式的多模态
                    // 注意：OpenAI 的 function_call 模式不支持在 tool result 中返回图片
                    // 因为 OpenAI API 要求 tool result 内容必须是字符串
                    if (capability.supportsImages || capability.supportsDocuments) {
                        // Gemini/Anthropic 支持在 functionResponse 中包含多模态数据
                        // 使用 displayName 而不是 name（Gemini API 支持 displayName）
                        const multimodalParts: ContentPart[] = multimodalData.map(item => ({
                            inlineData: {
                                mimeType: item.mimeType,
                                data: item.data,
                                displayName: item.name
                            }
                        }));
                        
                        // 从响应中移除 multimodal 数据（将放入 parts 中）
                        delete (response as any).multimodal;
                        
                        // 构建带 parts 的函数响应
                        responseParts.push({
                            functionResponse: {
                                name: call.name,
                                response,
                                id: call.id,
                                parts: multimodalParts
                            }
                        });
                        continue;
                    } else {
                        // 渠道不支持 function_call 模式的多模态（如 OpenAI）
                        // 对于 OpenAI function_call 模式，无法在 tool result 中返回图片
                        // 图片数据会被丢弃，AI 只能看到文件路径等文本信息
                        console.log(`[Multimodal] Channel ${channelType} does not support function_call multimodal, image data will be discarded`);
                        delete (response as any).multimodal;
                    }
                }
            }
            
            // 构建函数响应 part（包含 id 用于 Anthropic API）
            responseParts.push({
                functionResponse: {
                    name: call.name,
                    response,
                    id: call.id
                }
            });
        }
        
        // 在所有工具执行后创建一个检查点
        if (this.checkpointManager && conversationId !== undefined && messageIndex !== undefined) {
            const afterCheckpoint = await this.checkpointManager.createCheckpoint(
                conversationId,
                messageIndex,
                toolNameForCheckpoint,
                'after'
            );
            if (afterCheckpoint) {
                checkpoints.push(afterCheckpoint);
            }
        }
        
        return {
            responseParts,
            toolResults,
            checkpoints,
            multimodalAttachments: multimodalAttachments.length > 0 ? multimodalAttachments : undefined
        };
    }
    
    /**
     * 检查工具是否需要用户确认
     *
     * 使用统一的工具自动执行配置来判断
     * 如果工具被配置为自动执行（autoExec = true），则不需要确认
     * 如果工具被配置为需要确认（autoExec = false），则需要用户确认
     *
     * @param toolName 工具名称
     * @returns 是否需要确认
     */
    private toolNeedsConfirmation(toolName: string): boolean {
        if (!this.settingsManager) {
            return false;
        }
        
        // 使用统一的自动执行配置
        // isToolAutoExec 返回 true 表示自动执行，不需要确认
        // isToolAutoExec 返回 false 表示需要确认
        return !this.settingsManager.isToolAutoExec(toolName);
    }
    
    /**
     * 从函数调用列表中筛选出需要确认的工具
     */
    private getToolsNeedingConfirmation(
        calls: Array<{ name: string; args: Record<string, unknown>; id: string }>
    ): Array<{ name: string; args: Record<string, unknown>; id: string }> {
        return calls.filter(call => this.toolNeedsConfirmation(call.name));
    }
    
    /**
     * 处理工具确认响应
     *
     * 当用户在前端确认或拒绝工具执行时调用此方法
     *
     * @param request 工具确认响应数据
     */
    async *handleToolConfirmation(
        request: ToolConfirmationResponseData
    ): AsyncGenerator<
        ChatStreamChunkData | ChatStreamCompleteData | ChatStreamErrorData | ChatStreamToolIterationData | ChatStreamCheckpointsData | ChatStreamToolConfirmationData | ChatStreamToolsExecutingData
    > {
        try {
            const { conversationId, configId, toolResponses } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
                return;
            }
            
            // 3. 获取历史中最后一条 model 消息的函数调用
            const history = await this.conversationManager.getHistoryRef(conversationId);
            if (history.length === 0) {
                yield {
                    conversationId,
                    error: {
                        code: 'NO_HISTORY',
                        message: t('modules.api.chat.errors.noHistory')
                    }
                };
                return;
            }
            
            const lastMessage = history[history.length - 1];
            if (lastMessage.role !== 'model') {
                yield {
                    conversationId,
                    error: {
                        code: 'INVALID_STATE',
                        message: t('modules.api.chat.errors.lastMessageNotModel')
                    }
                };
                return;
            }
            
            const functionCalls = this.extractFunctionCalls(lastMessage);
            if (functionCalls.length === 0) {
                yield {
                    conversationId,
                    error: {
                        code: 'NO_FUNCTION_CALLS',
                        message: t('modules.api.chat.errors.noFunctionCalls')
                    }
                };
                return;
            }
            
            // 4. 分离确认和拒绝的工具调用
            const confirmedCalls = functionCalls.filter(call => {
                const response = toolResponses.find(r => r.id === call.id);
                return response?.confirmed;
            });
            const rejectedCalls = functionCalls.filter(call => {
                const response = toolResponses.find(r => r.id === call.id);
                return !response?.confirmed;
            });
            
            const messageIndex = history.length - 1;
            
            // 5. 执行确认的工具调用（复用 executeFunctionCallsWithResults）
            let confirmedResult: {
                responseParts: ContentPart[];
                toolResults: Array<{ id: string; name: string; result: Record<string, unknown> }>;
                checkpoints: CheckpointRecord[];
                multimodalAttachments?: ContentPart[];
            } = {
                responseParts: [],
                toolResults: [],
                checkpoints: []
            };
            
            if (confirmedCalls.length > 0) {
                // 工具执行前先发送计时信息（让前端立即显示）
                yield {
                    conversationId,
                    content: lastMessage,
                    toolsExecuting: true as const,
                    pendingToolCalls: confirmedCalls.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }))
                };
                
                confirmedResult = await this.executeFunctionCallsWithResults(
                    confirmedCalls,
                    conversationId,
                    messageIndex,
                    config,
                    request.abortSignal
                );
            }
            
            // 6. 处理拒绝的工具调用
            const rejectedParts: ContentPart[] = [];
            const rejectedResults: Array<{ id: string; name: string; result: Record<string, unknown> }> = [];
            
            for (const call of rejectedCalls) {
                const rejectionResponse = {
                    success: false,
                    error: t('modules.api.chat.errors.userRejectedTool'),
                    rejected: true
                };
                
                rejectedResults.push({
                    id: call.id,
                    name: call.name,
                    result: rejectionResponse
                });
                
                rejectedParts.push({
                    functionResponse: {
                        name: call.name,
                        response: rejectionResponse,
                        id: call.id
                    }
                });
            }
            
            // 7. 合并结果
            const allToolResults = [...confirmedResult.toolResults, ...rejectedResults];
            const allResponseParts = [...confirmedResult.responseParts, ...rejectedParts];
            const allCheckpoints = confirmedResult.checkpoints;
            
            // 8. 发送工具执行结果
            yield {
                conversationId,
                content: lastMessage,
                toolIteration: true as const,
                toolResults: allToolResults,
                checkpoints: allCheckpoints
            };
            
            // 9. 将函数响应添加到历史
            // 对于 XML/JSON 模式，如果有多模态附件，将其放在 parts 前面
            const confirmFunctionResponseParts = confirmedResult.multimodalAttachments && confirmedResult.multimodalAttachments.length > 0
                ? [...confirmedResult.multimodalAttachments, ...allResponseParts]
                : allResponseParts;
                
            await this.conversationManager.addContent(conversationId, {
                role: 'user',
                parts: confirmFunctionResponseParts,
                isFunctionResponse: true
            });
            
            // 7. 继续 AI 对话（让 AI 处理工具结果）
            // 获取动态系统提示词
            const dynamicSystemPrompt = this.promptManager.getSystemPrompt();
            
            // 工具调用循环
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;
                
                // 检查是否已取消
                if (request.abortSignal?.aborted) {
                    // 发送 cancelled 消息给前端，让前端正确清理状态
                    yield {
                        conversationId,
                        cancelled: true as const
                    } as any;
                    return;
                }
                
                // 获取对话历史
                const historyOptions = this.buildHistoryOptions(config);
                const updatedHistory = await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
                
                // 记录请求开始时间（用于计算响应持续时间）
                const confirmRequestStartTime = Date.now();
                
                // 调用 AI
                const response = await this.channelManager.generate({
                    configId,
                    history: updatedHistory,
                    abortSignal: request.abortSignal,
                    dynamicSystemPrompt
                });
                
                // 处理响应
                let finalContent: Content;
                
                if (this.isAsyncGenerator(response)) {
                    const accumulator = new StreamAccumulator();
                    // 设置请求开始时间，用于计算 responseDuration
                    accumulator.setRequestStartTime(confirmRequestStartTime);
                    accumulator.setProviderType(config.type as 'gemini' | 'openai' | 'anthropic' | 'custom');
                    let cancelled = false;
                    let lastPartsLength = 0;
                    
                    for await (const chunk of response) {
                        if (request.abortSignal?.aborted) {
                            cancelled = true;
                            break;
                        }
                        accumulator.add(chunk);
                        
                        const currentContent = accumulator.getContent();
                        const currentParts = currentContent.parts;
                        const newDelta = currentParts.slice(lastPartsLength);
                        lastPartsLength = currentParts.length;
                        
                        // 添加 thinkingStartTime 供前端实时显示
                        const confirmIterProcessedChunk: typeof chunk & { thinkingStartTime?: number } = {
                            ...chunk,
                            delta: newDelta.length > 0 ? newDelta : chunk.delta
                        };
                        
                        const confirmIterThinkingStartTime = currentContent.thinkingStartTime;
                        if (confirmIterThinkingStartTime !== undefined) {
                            confirmIterProcessedChunk.thinkingStartTime = confirmIterThinkingStartTime;
                        }
                        
                        yield { conversationId, chunk: confirmIterProcessedChunk };
                        if (chunk.done) {
                            break;
                        }
                    }
                    
                    if (cancelled) {
                        const partialContent = accumulator.getContent();
                        if (partialContent.parts.length > 0) {
                            await this.conversationManager.addContent(conversationId, partialContent);
                            // yield cancelled 消息，包含 partialContent 以便前端更新计时信息
                            yield {
                                conversationId,
                                cancelled: true as const,
                                content: partialContent
                            } as any;
                        }
                        return;
                    }
                    
                    finalContent = accumulator.getContent();
                } else {
                    finalContent = (response as GenerateResponse).content;
                    // 添加响应持续时间
                    finalContent.responseDuration = Date.now() - confirmRequestStartTime;
                    finalContent.chunkCount = 1;
                    yield {
                        conversationId,
                        chunk: {
                            delta: finalContent.parts,
                            done: true
                        }
                    };
                }
                
                // 转换工具调用格式
                this.convertXMLToolCallsToFunctionCalls(finalContent);
                this.ensureFunctionCallIds(finalContent);
                
                // 保存响应
                await this.conversationManager.addContent(conversationId, finalContent);
                
                // 检查是否有新的工具调用
                const newFunctionCalls = this.extractFunctionCalls(finalContent);
                
                if (newFunctionCalls.length === 0) {
                    // 没有工具调用，完成
                    yield {
                        conversationId,
                        content: finalContent
                    };
                    return;
                }
                
                // 检查新的工具调用是否需要确认
                const newToolsNeedingConfirmation = this.getToolsNeedingConfirmation(newFunctionCalls);
                
                if (newToolsNeedingConfirmation.length > 0) {
                    // 需要确认
                    const pendingToolCalls: PendingToolCall[] = newToolsNeedingConfirmation.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }));
                    
                    yield {
                        conversationId,
                        pendingToolCalls,
                        content: finalContent,
                        awaitingConfirmation: true as const
                    };
                    return;
                }
                
                // 不需要确认，直接执行
                const newHistory = await this.conversationManager.getHistoryRef(conversationId);
                const newMessageIndex = newHistory.length - 1;
                
                // 工具执行前先发送计时信息（让前端立即显示）
                yield {
                    conversationId,
                    content: finalContent,
                    toolsExecuting: true as const,
                    pendingToolCalls: newFunctionCalls.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }))
                };
                
                const newExecutionResult = await this.executeFunctionCallsWithResults(
                    newFunctionCalls,
                    conversationId,
                    newMessageIndex,
                    config,
                    request.abortSignal
                );
                
                // 先将函数响应添加到历史（确保取消时也能保存）
                // 对于 XML/JSON 模式，如果有多模态附件，将其放在 parts 前面
                const newFunctionResponseParts = newExecutionResult.multimodalAttachments
                    ? [...newExecutionResult.multimodalAttachments, ...newExecutionResult.responseParts]
                    : newExecutionResult.responseParts;
                
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: newFunctionResponseParts,
                    isFunctionResponse: true
                });
                
                // 检查是否有工具被取消
                const confirmHasCancelled = newExecutionResult.toolResults.some(r => (r.result as any).cancelled);
                if (confirmHasCancelled) {
                    // 有工具被取消，发送最终的 toolIteration 后结束
                    yield {
                        conversationId,
                        content: finalContent,
                        toolIteration: true as const,
                        toolResults: newExecutionResult.toolResults,
                        checkpoints: newExecutionResult.checkpoints
                    };
                    return;
                }
                
                yield {
                    conversationId,
                    content: finalContent,
                    toolIteration: true as const,
                    toolResults: newExecutionResult.toolResults,
                    checkpoints: newExecutionResult.checkpoints
                };
            }
            
            // 达到最大迭代次数
            yield {
                conversationId,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 检查是否是用户取消错误
            if (error instanceof ChannelError && error.type === ErrorType.CANCELLED_ERROR) {
                // 用户取消，yield cancelled 消息
                yield {
                    conversationId: request.conversationId,
                    cancelled: true as const
                } as any;
                return;
            }
            
            // 检查是否是取消导致的错误（信号已中止）
            if (request.abortSignal?.aborted) {
                yield {
                    conversationId: request.conversationId,
                    cancelled: true as const
                } as any;
            } else {
                yield {
                    conversationId: request.conversationId,
                    error: this.formatError(error)
                };
            }
        }
    }
    
    /**
     * 检查是否是 AsyncGenerator
     */
    private isAsyncGenerator(obj: any): obj is AsyncGenerator<StreamChunk> {
        return obj && typeof obj[Symbol.asyncIterator] === 'function';
    }
    
    /**
     * 识别对话中的回合
     *
     * 规则：
     * - 每个回合从一个非函数响应的用户消息开始
     * - 回合结束于下一个非函数响应的用户消息之前
     * - 每个回合记录该回合内最后一个助手消息的 totalTokenCount
     *
     * @param history 对话历史
     * @returns 回合列表
     */
    private identifyRounds(history: Content[]): ConversationRound[] {
        const rounds: ConversationRound[] = [];
        let currentRoundStart = -1;
        let currentRoundTokenCount: number | undefined;
        
        for (let i = 0; i < history.length; i++) {
            const message = history[i];
            
            if (message.role === 'user' && !message.isFunctionResponse) {
                // 找到一个非函数响应的用户消息，这是一个新回合的开始
                if (currentRoundStart !== -1) {
                    // 保存上一个回合
                    rounds.push({
                        startIndex: currentRoundStart,
                        endIndex: i,
                        tokenCount: currentRoundTokenCount
                    });
                }
                // 开始新回合
                currentRoundStart = i;
                currentRoundTokenCount = undefined;
            } else if (message.role === 'model') {
                // 记录助手消息的 token 数
                if (message.usageMetadata?.totalTokenCount !== undefined) {
                    currentRoundTokenCount = message.usageMetadata.totalTokenCount;
                }
            }
        }
        
        // 保存最后一个回合
        if (currentRoundStart !== -1) {
            rounds.push({
                startIndex: currentRoundStart,
                endIndex: history.length,
                tokenCount: currentRoundTokenCount
            });
        }
        
        return rounds;
    }
    
    /**
     * 计算上下文阈值
     *
     * @param threshold 阈值配置（数值或百分比字符串）
     * @param maxContextTokens 最大上下文 token 数
     * @returns 计算后的阈值
     */
    private calculateThreshold(threshold: number | string, maxContextTokens: number): number {
        if (typeof threshold === 'number') {
            return threshold;
        }
        
        // 百分比格式，如 "80%"
        if (threshold.endsWith('%')) {
            const percent = parseFloat(threshold.replace('%', ''));
            if (!isNaN(percent) && percent > 0 && percent <= 100) {
                return Math.floor(maxContextTokens * percent / 100);
            }
        }
        
        // 默认返回 80% 的最大上下文
        return Math.floor(maxContextTokens * 0.8);
    }
    
    /**
     * 计算上下文裁剪后应该从哪个索引开始获取历史
     *
     * 当最新助手消息的 totalTokenCount 超过阈值时，
     * 计算需要跳过的回合，返回应该开始的消息索引
     *
     * 注意：这个方法不删除任何消息，只是计算过滤的起始位置
     *
     * @param history 对话历史
     * @param config 渠道配置
     * @param latestTokenCount 最新助手消息的 totalTokenCount
     * @returns 应该开始获取历史的索引（0 表示不需要裁剪）
     */
    private calculateContextTrimStartIndex(
        history: Content[],
        config: BaseChannelConfig,
        latestTokenCount: number
    ): number {
        // 检查是否启用上下文阈值检测
        if (!config.contextThresholdEnabled) {
            return 0;
        }
        
        // 获取最大上下文和阈值
        const maxContextTokens = (config as any).maxContextTokens || 128000;
        const thresholdConfig = config.contextThreshold ?? '80%';
        const threshold = this.calculateThreshold(thresholdConfig, maxContextTokens);
        
        // 如果未超过阈值，无需裁剪
        if (latestTokenCount <= threshold) {
            return 0;
        }
        
        console.log(`[Context Trim] Token count ${latestTokenCount} exceeds threshold ${threshold}, calculating trim...`);
        
        // 识别回合
        const rounds = this.identifyRounds(history);
        
        // 至少需要保留当前回合（最后一个回合）
        if (rounds.length <= 1) {
            console.log('[Context Trim] Only one round, cannot trim');
            return 0;
        }
        
        // 估算每个回合的 token 数（基于最后一个有 token 记录的回合）
        // 简单策略：按回合数等比例估算
        const avgTokensPerRound = latestTokenCount / rounds.length;
        
        // 计算需要保留的回合数
        const targetTokens = threshold;
        const roundsToKeep = Math.max(1, Math.floor(targetTokens / avgTokensPerRound));
        
        // 需要跳过的回合数
        const roundsToSkip = Math.max(0, rounds.length - roundsToKeep);
        
        if (roundsToSkip === 0) {
            return 0;
        }
        
        // 返回应该开始的索引
        const startIndex = rounds[roundsToSkip].startIndex;
        console.log(`[Context Trim] Skipping ${roundsToSkip} rounds, starting from index ${startIndex}`);
        
        return startIndex;
    }
    
    /**
     * 构建 getHistoryForAPI 的选项
     * 根据渠道配置决定是否包含思考内容和签名
     */
    private buildHistoryOptions(config: BaseChannelConfig): GetHistoryOptions {
        // 检查是否启用了思考配置
        // Gemini 使用 options.thinkingConfig，Anthropic 使用 options 中的相关字段
        const thinkingEnabled = config.type === 'gemini'
            ? !!(config as any).optionsEnabled?.thinkingConfig
            : false;  // 其他提供商暂不支持
        
        // 获取多模态能力
        const channelType = (config.type || 'custom') as UtilChannelType;
        const toolMode = (config.toolMode || 'function_call') as UtilToolMode;
        const multimodalEnabled = config.multimodalToolsEnabled ?? false;
        const capability = getMultimodalCapability(channelType, toolMode, multimodalEnabled);
        
        return {
            // 当前轮次是否包含思考
            includeThoughts: thinkingEnabled,
            // 是否发送历史思考内容
            sendHistoryThoughts: config.sendHistoryThoughts ?? false,
            // 是否发送历史思考签名
            sendHistoryThoughtSignatures: config.sendHistoryThoughtSignatures ?? false,
            // 渠道类型，用于选择对应格式的签名
            channelType: config.type as 'gemini' | 'openai' | 'anthropic' | 'custom',
            // 多模态能力，用于过滤历史中的多模态数据
            multimodalCapability: capability
        };
    }
    
    /**
     * 查找历史中最后一个总结消息的索引
     *
     * @param history 对话历史
     * @returns 最后一个总结消息的索引，如果没有则返回 -1
     */
    private findLastSummaryIndex(history: Content[]): number {
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].isSummary) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * 获取用于 API 调用的历史，应用总结过滤和上下文阈值裁剪
     *
     * 优先级：
     * 1. 如果有总结消息，从最后一个总结消息开始获取历史
     * 2. 如果没有总结消息且 token 数超过阈值，从最旧的回合开始跳过
     *
     * @param conversationId 对话 ID
     * @param config 渠道配置
     * @param historyOptions 历史选项
     * @returns 裁剪后的历史
     */
    private async getHistoryWithContextTrim(
        conversationId: string,
        config: BaseChannelConfig,
        historyOptions: GetHistoryOptions
    ): Promise<Content[]> {
        // 先获取完整的原始历史
        const fullHistory = await this.conversationManager.getHistoryRef(conversationId);
        
        // 如果历史为空，直接返回
        if (fullHistory.length === 0) {
            return [];
        }
        
        // 查找最后一个总结消息
        const lastSummaryIndex = this.findLastSummaryIndex(fullHistory);
        
        // 如果有总结消息，从总结消息开始获取历史
        if (lastSummaryIndex >= 0) {
            console.log(`[Context] Found summary at index ${lastSummaryIndex}, using as starting point`);
            
            // 获取完整的 API 历史
            const fullApiHistory = await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
            
            // 计算对应的 API 历史起始索引
            // 由于 API 历史可能过滤了一些内容，需要找到对应位置
            const trimRatio = lastSummaryIndex / fullHistory.length;
            const apiStartIndex = Math.floor(fullApiHistory.length * trimRatio);
            
            return fullApiHistory.slice(apiStartIndex);
        }
        
        // 没有总结消息，检查上下文阈值
        // 查找最新的 model 消息，获取 totalTokenCount
        let latestTokenCount: number | undefined;
        for (let i = fullHistory.length - 1; i >= 0; i--) {
            if (fullHistory[i].role === 'model' && fullHistory[i].usageMetadata?.totalTokenCount) {
                latestTokenCount = fullHistory[i].usageMetadata!.totalTokenCount;
                break;
            }
        }
        
        // 如果没有 token 记录，直接获取完整历史
        if (latestTokenCount === undefined) {
            return await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
        }
        
        // 计算需要跳过的起始索引
        const startIndex = this.calculateContextTrimStartIndex(fullHistory, config, latestTokenCount);
        
        // 如果不需要裁剪，直接返回完整历史
        if (startIndex === 0) {
            return await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
        }
        
        // 获取完整的 API 历史
        const fullApiHistory = await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
        
        // 裁剪历史：从 startIndex 开始
        const trimRatio = startIndex / fullHistory.length;
        const apiStartIndex = Math.floor(fullApiHistory.length * trimRatio);
        
        const trimmedHistory = fullApiHistory.slice(apiStartIndex);
        
        // 确保历史以 user 消息开始（Gemini API 要求）
        if (trimmedHistory.length > 0 && trimmedHistory[0].role !== 'user') {
            const firstUserIndex = trimmedHistory.findIndex(m => m.role === 'user');
            if (firstUserIndex > 0) {
                return trimmedHistory.slice(firstUserIndex);
            }
        }
        
        return trimmedHistory;
    }
    
    /**
     * 处理上下文总结请求
     *
     * 将指定范围的对话历史压缩为一条总结消息
     *
     * @param request 总结请求数据
     * @returns 总结响应数据
     */
    async handleSummarizeContext(
        request: SummarizeContextRequestData
    ): Promise<SummarizeContextSuccessData | SummarizeContextErrorData> {
        try {
            const { conversationId, configId } = request;
            
            // 从设置中读取总结配置
            let configKeepRecentRounds = 2;  // 默认值
            let configSummarizePrompt = '';  // 默认值（空则使用内置提示词）
            let useSeparateModel = false;
            let summarizeChannelId = '';
            let summarizeModelId = '';
            
            if (this.settingsManager) {
                const summarizeConfig = this.settingsManager.getSummarizeConfig();
                if (summarizeConfig) {
                    if (typeof summarizeConfig.keepRecentRounds === 'number') {
                        configKeepRecentRounds = summarizeConfig.keepRecentRounds;
                    }
                    if (typeof summarizeConfig.summarizePrompt === 'string') {
                        configSummarizePrompt = summarizeConfig.summarizePrompt;
                    }
                    useSeparateModel = !!summarizeConfig.useSeparateModel;
                    summarizeChannelId = summarizeConfig.summarizeChannelId || '';
                    summarizeModelId = summarizeConfig.summarizeModelId || '';
                }
            }
            const keepRecentRounds = configKeepRecentRounds;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 确定使用的渠道配置
            // 如果启用了专用总结模型且配置了渠道，使用总结专用渠道
            let actualConfigId = configId;
            let actualModelId: string | undefined;
            
            if (useSeparateModel && summarizeChannelId) {
                // 验证专用总结渠道配置
                const summarizeConfig = await this.configManager.getConfig(summarizeChannelId);
                if (summarizeConfig && summarizeConfig.enabled) {
                    actualConfigId = summarizeChannelId;
                    if (summarizeModelId) {
                        actualModelId = summarizeModelId;
                    }
                    console.log(`[Summarize] Using dedicated model: channel=${summarizeChannelId}, model=${summarizeModelId || 'default'}`);
                } else {
                    console.log(`[Summarize] Dedicated channel not available, falling back to chat config`);
                }
            }
            
            // 3. 验证配置
            const config = await this.configManager.getConfig(actualConfigId);
            if (!config) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId: actualConfigId })
                    }
                };
            }
            
            if (!config.enabled) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId: actualConfigId })
                    }
                };
            }
            
            // 3. 获取对话历史
            const fullHistory = await this.conversationManager.getHistoryRef(conversationId);
            
            // 4. 找到最后一个总结消息的位置，从该位置之后开始识别回合
            const lastSummaryIndex = this.findLastSummaryIndex(fullHistory);
            const historyStartIndex = lastSummaryIndex >= 0 ? lastSummaryIndex + 1 : 0;
            
            // 只对总结之后的历史进行回合识别
            const historyAfterSummary = fullHistory.slice(historyStartIndex);
            const rounds = this.identifyRounds(historyAfterSummary);
            
            if (rounds.length <= keepRecentRounds) {
                return {
                    success: false,
                    error: {
                        code: 'NOT_ENOUGH_ROUNDS',
                        message: t('modules.api.chat.errors.notEnoughRounds', { currentRounds: rounds.length, keepRounds: keepRecentRounds })
                    }
                };
            }
            
            // 5. 确定总结范围
            // 总结从总结后开始到 (rounds.length - keepRecentRounds) 回合
            const roundsToSummarize = rounds.length - keepRecentRounds;
            
            // 边界检查：确保有内容需要总结
            if (roundsToSummarize <= 0) {
                return {
                    success: false,
                    error: {
                        code: 'NOT_ENOUGH_CONTENT',
                        message: t('modules.api.chat.errors.notEnoughContent', { currentRounds: rounds.length, keepRounds: keepRecentRounds })
                    }
                };
            }
            
            // 计算总结范围的结束索引（相对于 fullHistory）
            // 如果 roundsToSummarize == rounds.length（即 keepRecentRounds == 0），则总结所有回合
            const summarizeEndIndexRelative = roundsToSummarize >= rounds.length
                ? historyAfterSummary.length
                : rounds[roundsToSummarize].startIndex;
            const summarizeEndIndex = historyStartIndex + summarizeEndIndexRelative;
            
            // 提取需要总结的消息（包括之前的总结消息，便于AI理解上下文）
            const messagesToSummarize = fullHistory.slice(0, summarizeEndIndex);
            
            if (messagesToSummarize.length === 0) {
                return {
                    success: false,
                    error: {
                        code: 'NO_MESSAGES_TO_SUMMARIZE',
                        message: t('modules.api.chat.errors.noMessagesToSummarize')
                    }
                };
            }
            
            // 6. 构建总结请求（直接输出总结内容）
            const defaultPrompt = t('modules.api.chat.prompts.defaultSummarizePrompt');
            // 使用配置中的提示词，如果没有则使用默认提示词
            const prompt = configSummarizePrompt || defaultPrompt;
            
            // 构建历史（需要总结的完整消息 + 总结请求）
            // 保留完整历史，让 AI 理解上下文
            const summaryRequestHistory: Content[] = [
                ...messagesToSummarize,
                {
                    role: 'user',
                    parts: [{ text: prompt }]
                }
            ];
            
            // 7. 调用 AI 生成总结（不传递工具，不重试）
            // 使用确定的渠道配置和可能的专用模型
            const generateOptions: {
                configId: string;
                history: Content[];
                abortSignal?: AbortSignal;
                skipTools: boolean;
                skipRetry: boolean;
                modelOverride?: string;
            } = {
                configId: actualConfigId,
                history: summaryRequestHistory,
                abortSignal: request.abortSignal,
                skipTools: true,   // 不传递工具声明
                skipRetry: true    // 总结操作不进行重试
            };
            
            // 如果指定了专用模型，添加模型覆盖
            if (actualModelId) {
                generateOptions.modelOverride = actualModelId;
            }
            
            const response = await this.channelManager.generate(generateOptions);
            
            // 处理响应（可能是流式或非流式）
            let finalContent: Content;
            
            if (this.isAsyncGenerator(response)) {
                // 流式响应：累积内容
                const accumulator = new StreamAccumulator();
                accumulator.setProviderType(config.type as 'gemini' | 'openai' | 'anthropic' | 'custom');
                
                for await (const chunk of response) {
                    if (request.abortSignal?.aborted) {
                        return {
                            success: false,
                            error: {
                                code: 'ABORTED',
                                message: t('modules.api.chat.errors.summarizeAborted')
                            }
                        };
                    }
                    accumulator.add(chunk);
                    if (chunk.done) {
                        break;
                    }
                }
                
                finalContent = accumulator.getContent();
            } else {
                // 非流式响应
                finalContent = (response as GenerateResponse).content;
            }
            
            // 8. 提取 token 信息
            const beforeTokenCount = finalContent.usageMetadata?.promptTokenCount;
            const afterTokenCount = finalContent.usageMetadata?.candidatesTokenCount;
            
            // 9. 提取总结文本（直接使用响应内容）
            const summaryText = finalContent.parts
                .filter(p => p.text && !p.thought)
                .map(p => p.text)
                .join('\n')
                .trim();
            
            if (!summaryText) {
                return {
                    success: false,
                    error: {
                        code: 'EMPTY_SUMMARY',
                        message: t('modules.api.chat.errors.emptySummary')
                    }
                };
            }
            
            // 10. 删除已存在的旧总结消息（如果有）
            // 检查 summarizeEndIndex 位置及之前是否有总结消息
            let insertIndex = summarizeEndIndex;
            const currentHistory = await this.conversationManager.getHistoryRef(conversationId);
            
            // 从 summarizeEndIndex 往前找所有的 isSummary 消息并删除
            // 因为新总结会覆盖之前所有总结的内容
            const summaryIndicesToDelete: number[] = [];
            for (let i = 0; i < summarizeEndIndex; i++) {
                if (currentHistory[i]?.isSummary) {
                    summaryIndicesToDelete.push(i);
                }
            }
            
            // 从后往前删除旧总结（避免索引偏移问题）
            if (summaryIndicesToDelete.length > 0) {
                // 删除旧总结消息
                for (let i = summaryIndicesToDelete.length - 1; i >= 0; i--) {
                    const indexToDelete = summaryIndicesToDelete[i];
                    await this.conversationManager.deleteMessage(conversationId, indexToDelete);
                }
                // 调整插入位置（因为删除了一些消息）
                insertIndex = summarizeEndIndex - summaryIndicesToDelete.length;
            }
            
            // 11. 创建总结消息并添加到历史
            const summaryContent: Content = {
                role: 'user',
                parts: [{ text: `${t('modules.api.chat.prompts.summaryPrefix')}\n\n${summaryText}` }],
                isSummary: true,
                summarizedMessageCount: messagesToSummarize.length,
                // 保存 token 信息用于前端显示
                usageMetadata: {
                    promptTokenCount: beforeTokenCount,
                    candidatesTokenCount: afterTokenCount
                }
            };
            
            // 在调整后的位置插入总结消息
            await this.conversationManager.insertContent(conversationId, insertIndex, summaryContent);
            
            return {
                success: true,
                summaryContent,
                summarizedMessageCount: messagesToSummarize.length,
                beforeTokenCount,
                afterTokenCount
            };
            
        } catch (error) {
            return {
                success: false,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 处理重试请求（非流式）
     * 支持工具调用循环
     *
     * @param request 重试请求数据
     * @returns 对话响应数据
     */
    async handleRetry(request: RetryRequestData): Promise<ChatSuccessData | ChatErrorData> {
        try {
            const { conversationId, configId } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
            }
            
            if (!config.enabled) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
            }
            
            // 3. 工具调用循环
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;
                
                // 获取对话历史
                const historyOptions = this.buildHistoryOptions(config);
                const history = await this.conversationManager.getHistoryForAPI(conversationId, historyOptions);
                
                // 调用 AI（非流式）
                const response = await this.channelManager.generate({
                    configId,
                    history
                });
                
                // 类型守卫：确保是 GenerateResponse
                if (!('content' in response)) {
                    throw new Error('Unexpected stream response from generate()');
                }
                
                const generateResponse = response as GenerateResponse;
                
                // 为没有 id 的 functionCall 添加唯一 id（Gemini 格式不返回 id）
                this.ensureFunctionCallIds(generateResponse.content);
                
                // 为没有 id 的 functionCall 添加唯一 id
                this.ensureFunctionCallIds(generateResponse.content);
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, generateResponse.content);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(generateResponse.content);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，结束循环并返回
                    return {
                        success: true,
                        content: generateResponse.content
                    };
                }
                
                // 有工具调用，执行工具并添加结果
                // 获取当前消息索引
                const retryHistory = await this.conversationManager.getHistoryRef(conversationId);
                const retryMessageIndex = retryHistory.length - 1;
                
                const functionResponses = await this.executeFunctionCalls(functionCalls, conversationId, retryMessageIndex);
                
                // 将函数响应添加到历史（作为 user 消息，标记为函数响应）
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: functionResponses,
                    isFunctionResponse: true
                });
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            return {
                success: false,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 错误处理
            return {
                success: false,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 处理重试请求（自动根据配置决定使用流式或非流式）
     * 支持工具调用循环
     *
     * @param request 重试请求数据
     * @returns 异步生成器，产生流式响应
     */
    async *handleRetryStream(
        request: RetryRequestData
    ): AsyncGenerator<
        ChatStreamChunkData | ChatStreamCompleteData | ChatStreamErrorData | ChatStreamToolIterationData | ChatStreamToolConfirmationData | ChatStreamToolsExecutingData
    > {
        try {
            const { conversationId, configId } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
                return;
            }
            
            if (!config.enabled) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
                return;
            }
            
            // 3. 中断之前未完成的 diff 等待
            const retryDiffManager = getDiffManager();
            retryDiffManager.markUserInterrupt();
            
            // 4. 检查并处理孤立的函数调用
            // 如果历史最后一条 model 消息包含 functionCall 但没有对应的 functionResponse
            // 则先执行这些函数调用
            const orphanedFunctionCalls = await this.checkAndExecuteOrphanedFunctionCalls(conversationId);
            if (orphanedFunctionCalls) {
                // 发送孤立函数调用的执行结果到前端
                yield {
                    conversationId,
                    content: orphanedFunctionCalls.functionCallContent,
                    toolIteration: true as const,
                    toolResults: orphanedFunctionCalls.toolResults
                };
            }
            
            // 5. 重置中断标记
            retryDiffManager.resetUserInterrupt();
            
            // 6. 判断是否需要刷新动态系统提示词
            // 如果是重试首条消息之后的回复（历史只有1条用户消息），需要刷新
            const retryHistoryCheck = await this.conversationManager.getHistoryRef(conversationId);
            const isRetryFirstMessage = retryHistoryCheck.length === 1 && retryHistoryCheck[0].role === 'user';
            
            const retryDynamicPrompt = isRetryFirstMessage
                ? this.promptManager.refreshAndGetPrompt()
                : this.promptManager.getSystemPrompt();
            
            // 7. 工具调用循环
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;

                // 检查是否已取消
                if (request.abortSignal?.aborted) {
                    // 发送 cancelled 消息给前端，让前端正确清理状态
                    yield {
                        conversationId,
                        cancelled: true as const
                    } as any;
                    return;
                }
 
                // 获取对话历史
                const retryStreamHistoryOptions = this.buildHistoryOptions(config);
                const history = await this.conversationManager.getHistoryForAPI(conversationId, retryStreamHistoryOptions);
                
                // 记录请求开始时间（用于计算响应持续时间）
                const requestStartTime = Date.now();
                
                // 调用 AI（传递 abortSignal 和动态系统提示词）
                const response = await this.channelManager.generate({
                    configId,
                    history,
                    abortSignal: request.abortSignal,
                    dynamicSystemPrompt: retryDynamicPrompt
                });
                
                // 处理响应
                let finalContent: Content;
                
                if (this.isAsyncGenerator(response)) {
                    // 流式响应，累加器会自动从全局设置获取 toolMode
                    const accumulator = new StreamAccumulator();
                    // 设置请求开始时间，用于计算 responseDuration
                    accumulator.setRequestStartTime(requestStartTime);
                    // 根据配置类型设置 providerType（用于多格式思考签名存储）
                    accumulator.setProviderType(config.type as 'gemini' | 'openai' | 'anthropic' | 'custom');
                    let cancelled = false;
                    let lastPartsLength = 0;
                    
                    try {
                    for await (const chunk of response) {
                        // 检查是否已取消
                        if (request.abortSignal?.aborted) {
                            cancelled = true;
                            break;
                        }
                        accumulator.add(chunk);
                        
                        // 获取累加器处理后的 parts（实时转换 JSON 工具调用标记）
                        const currentContent = accumulator.getContent();
                        const currentParts = currentContent.parts;
                        
                        // 计算增量 delta（新增的 parts）
                        const newDelta = currentParts.slice(lastPartsLength);
                        lastPartsLength = currentParts.length;
                        
                        // 如果有修改过的 parts（如工具调用转换），需要发送更新
                        // 这里简化处理：直接发送累加器的增量
                        // 添加 thinkingStartTime 供前端实时显示
                        const retryProcessedChunk: typeof chunk & { thinkingStartTime?: number } = {
                            ...chunk,
                            delta: newDelta.length > 0 ? newDelta : chunk.delta
                        };
                        
                        const retryThinkingStartTime = currentContent.thinkingStartTime;
                        if (retryThinkingStartTime !== undefined) {
                            retryProcessedChunk.thinkingStartTime = retryThinkingStartTime;
                        }
                        
                        yield { conversationId, chunk: retryProcessedChunk };
                        if (chunk.done) {
                            break;
                        }
                    }
                    
                    // 如果已取消，保存已接收的内容并 yield cancelled 消息
                    if (cancelled) {
                        const partialContent = accumulator.getContent();
                        if (partialContent.parts.length > 0) {
                            await this.conversationManager.addContent(conversationId, partialContent);
                            // yield cancelled 消息，包含 partialContent 以便前端更新计时信息
                            yield {
                                conversationId,
                                cancelled: true as const,
                                content: partialContent
                            } as any;
                        } else {
                            yield {
                                conversationId,
                                cancelled: true as const
                            } as any;
                        }
                        return;
                    }
                    
                    finalContent = accumulator.getContent();
                    } catch (streamError) {
                        throw streamError;
                    }
                } else {
                    // 非流式响应
                    finalContent = (response as GenerateResponse).content;
                    // 添加响应持续时间
                    finalContent.responseDuration = Date.now() - requestStartTime;
                    finalContent.chunkCount = 1;
                    yield {
                        conversationId,
                        chunk: {
                            delta: finalContent.parts,
                            done: true
                        }
                    };
                }
                
                // 转换 XML 工具调用为 functionCall 格式（如果有）
                this.convertXMLToolCallsToFunctionCalls(finalContent);
                
                // 为没有 id 的 functionCall 添加唯一 id
                this.ensureFunctionCallIds(finalContent);
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, finalContent);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(finalContent);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，为模型消息创建存档点（如果配置了执行后）
                    const retryModelMessageCheckpoints: CheckpointRecord[] = [];
                    if (this.checkpointManager && this.settingsManager?.shouldCreateAfterModelMessageCheckpoint()) {
                        const modelHistory = await this.conversationManager.getHistoryRef(conversationId);
                        const modelMessageIndex = modelHistory.length - 1;
                        
                        const checkpoint = await this.checkpointManager.createCheckpoint(
                            conversationId,
                            modelMessageIndex,
                            'model_message',
                            'after'
                        );
                        if (checkpoint) {
                            retryModelMessageCheckpoints.push(checkpoint);
                        }
                    }
                    
                    // 结束循环，返回完成数据（重试时只有模型消息的检查点）
                    yield {
                        conversationId,
                        content: finalContent,
                        checkpoints: retryModelMessageCheckpoints
                    };
                    return;
                }
                
                // 有工具调用，检查是否需要确认
                const retryToolsNeedingConfirmation = this.getToolsNeedingConfirmation(functionCalls);
                
                if (retryToolsNeedingConfirmation.length > 0) {
                    // 有工具需要确认，发送确认请求到前端
                    const pendingToolCalls: PendingToolCall[] = retryToolsNeedingConfirmation.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }));
                    
                    yield {
                        conversationId,
                        pendingToolCalls,
                        content: finalContent,
                        awaitingConfirmation: true as const
                    };
                    
                    // 暂停执行，等待前端调用 handleToolConfirmation
                    return;
                }
                
                // 不需要确认，直接执行工具
                // 获取当前消息索引
                const retryStreamHistory = await this.conversationManager.getHistoryRef(conversationId);
                const retryStreamMessageIndex = retryStreamHistory.length - 1;
                
                // 工具执行前先发送计时信息（让前端立即显示）
                yield {
                    conversationId,
                    content: finalContent,
                    toolsExecuting: true as const,
                    pendingToolCalls: functionCalls.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }))
                };
                
                const { responseParts, toolResults, checkpoints, multimodalAttachments: retryMultimodalAttachments } = await this.executeFunctionCallsWithResults(
                    functionCalls,
                    conversationId,
                    retryStreamMessageIndex,
                    config,
                    request.abortSignal
                );
                
                // 先将函数响应添加到历史（确保取消时也能保存）
                // 对于 XML/JSON 模式，如果有多模态附件，将其放在 parts 前面
                const retryFunctionResponseParts = retryMultimodalAttachments
                    ? [...retryMultimodalAttachments, ...responseParts]
                    : responseParts;
                    
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: retryFunctionResponseParts,
                    isFunctionResponse: true
                });
                
                // 检查是否有工具被取消
                const retryHasCancelled = toolResults.some(r => (r.result as any).cancelled);
                if (retryHasCancelled) {
                    // 有工具被取消，发送最终的 toolIteration 后结束
                    yield {
                        conversationId,
                        content: finalContent,
                        toolIteration: true as const,
                        toolResults,
                        checkpoints
                    };
                    return;
                }
                
                // 发送 toolIteration 信号（包含检查点）
                yield {
                    conversationId,
                    content: finalContent,
                    toolIteration: true as const,
                    toolResults,
                    checkpoints
                };
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            yield {
                conversationId,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 检查是否是用户取消错误
            if (error instanceof ChannelError && error.type === ErrorType.CANCELLED_ERROR) {
                // 用户取消，yield cancelled 消息
                yield {
                    conversationId: request.conversationId,
                    cancelled: true as const
                } as any;
                return;
            }
            
            yield {
                conversationId: request.conversationId,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 处理编辑并重试请求（非流式）
     * 支持工具调用循环
     *
     * @param request 编辑并重试请求数据
     * @returns 对话响应数据
     */
    async handleEditAndRetry(
        request: EditAndRetryRequestData
    ): Promise<ChatSuccessData | ChatErrorData> {
        try {
            const { conversationId, messageIndex, newMessage, configId } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
            }
            
            if (!config.enabled) {
                return {
                    success: false,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
            }
            
            // 3. 验证消息索引和角色
            const message = await this.conversationManager.getMessage(conversationId, messageIndex);
            if (!message) {
                return {
                    success: false,
                    error: {
                        code: 'MESSAGE_NOT_FOUND',
                        message: t('modules.api.chat.errors.messageNotFound', { messageIndex })
                    }
                };
            }
            
            if (message.role !== 'user') {
                return {
                    success: false,
                    error: {
                        code: 'INVALID_MESSAGE_ROLE',
                        message: t('modules.api.chat.errors.canOnlyEditUserMessage', { role: message.role })
                    }
                };
            }
            
            // 4. 更新消息内容
            await this.conversationManager.updateMessage(conversationId, messageIndex, {
                parts: [{ text: newMessage }]
            });
            
            // 5. 删除后续所有消息（messageIndex+1 及之后）和关联的检查点
            const historyRef = await this.conversationManager.getHistoryRef(conversationId);
            if (messageIndex + 1 < historyRef.length) {
                // 先删除检查点
                if (this.checkpointManager) {
                    await this.checkpointManager.deleteCheckpointsFromIndex(conversationId, messageIndex + 1);
                }
                await this.conversationManager.deleteToMessage(conversationId, messageIndex + 1);
            }
            
            // 6. 工具调用循环
            const editHistoryOptions = this.buildHistoryOptions(config);
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;
                
                // 获取更新后的历史
                const updatedHistory = await this.conversationManager.getHistoryForAPI(conversationId, editHistoryOptions);
                
                // 调用 AI（非流式）
                const response = await this.channelManager.generate({
                    configId,
                    history: updatedHistory
                });
                
                // 类型守卫：确保是 GenerateResponse
                if (!('content' in response)) {
                    throw new Error('Unexpected stream response from generate()');
                }
                
                const generateResponse = response as GenerateResponse;
                
                // 为没有 id 的 functionCall 添加唯一 id
                this.ensureFunctionCallIds(generateResponse.content);
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, generateResponse.content);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(generateResponse.content);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，结束循环并返回
                    return {
                        success: true,
                        content: generateResponse.content
                    };
                }
                
                // 有工具调用，执行工具并添加结果
                // 获取当前消息索引
                const editHistory = await this.conversationManager.getHistoryRef(conversationId);
                const editMessageIndex = editHistory.length - 1;
                
                const functionResponses = await this.executeFunctionCalls(functionCalls, conversationId, editMessageIndex);
                
                // 将函数响应添加到历史（作为 user 消息，标记为函数响应）
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: functionResponses,
                    isFunctionResponse: true
                });
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            return {
                success: false,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 错误处理
            return {
                success: false,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 处理编辑并重试请求（自动根据配置决定使用流式或非流式）
     * 支持工具调用循环
     *
     * @param request 编辑并重试请求数据
     * @returns 异步生成器，产生流式响应
     */
    async *handleEditAndRetryStream(
        request: EditAndRetryRequestData
    ): AsyncGenerator<
        ChatStreamChunkData | ChatStreamCompleteData | ChatStreamErrorData | ChatStreamToolIterationData | ChatStreamCheckpointsData | ChatStreamToolConfirmationData | ChatStreamToolsExecutingData
    > {
        try {
            const { conversationId, messageIndex, newMessage, configId } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 验证配置
            const config = await this.configManager.getConfig(configId);
            if (!config) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_NOT_FOUND',
                        message: t('modules.api.chat.errors.configNotFound', { configId })
                    }
                };
                return;
            }
            
            if (!config.enabled) {
                yield {
                    conversationId,
                    error: {
                        code: 'CONFIG_DISABLED',
                        message: t('modules.api.chat.errors.configDisabled', { configId })
                    }
                };
                return;
            }
            
            // 3. 验证消息索引和角色
            const message = await this.conversationManager.getMessage(conversationId, messageIndex);
            if (!message) {
                yield {
                    conversationId,
                    error: {
                        code: 'MESSAGE_NOT_FOUND',
                        message: t('modules.api.chat.errors.messageNotFound', { messageIndex })
                    }
                };
                return;
            }
            
            if (message.role !== 'user') {
                yield {
                    conversationId,
                    error: {
                        code: 'INVALID_MESSAGE_ROLE',
                        message: t('modules.api.chat.errors.canOnlyEditUserMessage', { role: message.role })
                    }
                };
                return;
            }
            
            // 4. 中断之前未完成的 diff 等待
            const editDiffManager = getDiffManager();
            editDiffManager.markUserInterrupt();
            
            // 5. 删除该消息及后续所有消息的检查点（因为编辑会改变内容，原检查点无效）
            if (this.checkpointManager) {
                await this.checkpointManager.deleteCheckpointsFromIndex(conversationId, messageIndex);
            }
            
            // 6. 为编辑后的用户消息创建存档点（执行前）
            if (this.checkpointManager && this.settingsManager?.shouldCreateBeforeUserMessageCheckpoint()) {
                const checkpoint = await this.checkpointManager.createCheckpoint(
                    conversationId,
                    messageIndex,
                    'user_message',
                    'before'
                );
                if (checkpoint) {
                    yield {
                        conversationId,
                        checkpoints: [checkpoint],
                        checkpointOnly: true as const
                    };
                }
            }
            
            // 7. 更新消息内容（包含附件）
            const editParts = this.buildUserMessageParts(newMessage, request.attachments);
            await this.conversationManager.updateMessage(conversationId, messageIndex, {
                parts: editParts
            });
            
            // 8. 删除后续所有消息
            const historyRef = await this.conversationManager.getHistoryRef(conversationId);
            if (messageIndex + 1 < historyRef.length) {
                await this.conversationManager.deleteToMessage(conversationId, messageIndex + 1);
            }
            
            // 9. 为编辑后的用户消息创建存档点（执行后）
            if (this.checkpointManager && this.settingsManager?.shouldCreateAfterUserMessageCheckpoint()) {
                const checkpoint = await this.checkpointManager.createCheckpoint(
                    conversationId,
                    messageIndex,
                    'user_message',
                    'after'
                );
                if (checkpoint) {
                    yield {
                        conversationId,
                        checkpoints: [checkpoint],
                        checkpointOnly: true as const
                    };
                }
            }
            
            // 10. 重置中断标记
            editDiffManager.resetUserInterrupt();
            
            // 11. 判断是否是编辑首条消息（需要刷新动态系统提示词）
            const isEditFirstMessage = messageIndex === 0;
            const editDynamicPrompt = isEditFirstMessage
                ? this.promptManager.refreshAndGetPrompt()
                : this.promptManager.getSystemPrompt();
            
            // 12. 工具调用循环
            let iteration = 0;
            while (iteration < MAX_TOOL_CALL_ITERATIONS) {
                iteration++;
                
                // 检查是否已取消
                if (request.abortSignal?.aborted) {
                    // 发送 cancelled 消息给前端，让前端正确清理状态
                    yield {
                        conversationId,
                        cancelled: true as const
                    } as any;
                    return;
                }

                // 为模型消息创建存档点（如果配置了执行前）
                // 根据 modelOuterLayerOnly 设置决定是否在每次迭代都创建
                const editOuterLayerOnly = this.settingsManager?.isModelOuterLayerOnly() ?? true;
                const shouldCreateEditBeforeCheckpoint = this.checkpointManager &&
                    this.settingsManager?.shouldCreateBeforeModelMessageCheckpoint() &&
                    (!editOuterLayerOnly || iteration === 1);  // 只在最外层模式下，只在第一次迭代创建
                
                if (shouldCreateEditBeforeCheckpoint) {
                    const currentHistoryBeforeModel = await this.conversationManager.getHistoryRef(conversationId);
                    const modelMessageIndexBefore = currentHistoryBeforeModel.length;
                    
                    const checkpoint = await this.checkpointManager!.createCheckpoint(
                        conversationId,
                        modelMessageIndexBefore,
                        'model_message',
                        'before'
                    );
                    if (checkpoint) {
                        yield {
                            conversationId,
                            checkpoints: [checkpoint],
                            checkpointOnly: true as const
                        };
                    }
                }
                
                // 获取更新后的历史
                const editStreamHistoryOptions = this.buildHistoryOptions(config);
                const updatedHistory = await this.conversationManager.getHistoryForAPI(conversationId, editStreamHistoryOptions);
                
                // 记录请求开始时间（用于计算响应持续时间）
                const editRequestStartTime = Date.now();
                
                // 调用 AI（传递 abortSignal 和动态系统提示词）
                const response = await this.channelManager.generate({
                    configId,
                    history: updatedHistory,
                    abortSignal: request.abortSignal,
                    dynamicSystemPrompt: editDynamicPrompt
                });
                
                // 处理响应
                let finalContent: Content;
                
                if (this.isAsyncGenerator(response)) {
                    // 流式响应，累加器会自动从全局设置获取 toolMode
                    const accumulator = new StreamAccumulator();
                    // 设置请求开始时间，用于计算 responseDuration
                    accumulator.setRequestStartTime(editRequestStartTime);
                    // 根据配置类型设置 providerType（用于多格式思考签名存储）
                    accumulator.setProviderType(config.type as 'gemini' | 'openai' | 'anthropic' | 'custom');
                    let cancelled = false;
                    let lastPartsLengthEdit = 0;
                    
                    for await (const chunk of response) {
                        // 检查是否已取消
                        if (request.abortSignal?.aborted) {
                            cancelled = true;
                            break;
                        }
                        accumulator.add(chunk);
                        
                        // 获取累加器处理后的 parts
                        const currentContent = accumulator.getContent();
                        const currentParts = currentContent.parts;
                        
                        // 计算增量 delta
                        const newDeltaEdit = currentParts.slice(lastPartsLengthEdit);
                        lastPartsLengthEdit = currentParts.length;
                        
                        // 发送处理后的 chunk，添加 thinkingStartTime 供前端实时显示
                        const processedChunkEdit: typeof chunk & { thinkingStartTime?: number } = {
                            ...chunk,
                            delta: newDeltaEdit.length > 0 ? newDeltaEdit : chunk.delta
                        };
                        
                        const editThinkingStartTime = currentContent.thinkingStartTime;
                        if (editThinkingStartTime !== undefined) {
                            processedChunkEdit.thinkingStartTime = editThinkingStartTime;
                        }
                        
                        yield { conversationId, chunk: processedChunkEdit };
                        if (chunk.done) {
                            break;
                        }
                    }
                    
                    // 如果已取消，保存已接收的内容并返回
                    if (cancelled) {
                        const partialContent = accumulator.getContent();
                        if (partialContent.parts.length > 0) {
                            await this.conversationManager.addContent(conversationId, partialContent);
                            // yield cancelled 消息，包含 partialContent 以便前端更新计时信息
                            yield {
                                conversationId,
                                cancelled: true as const,
                                content: partialContent
                            } as any;
                        }
                        return;
                    }
                    
                    finalContent = accumulator.getContent();
                } else {
                    // 非流式响应
                    finalContent = (response as GenerateResponse).content;
                    // 添加响应持续时间
                    finalContent.responseDuration = Date.now() - editRequestStartTime;
                    finalContent.chunkCount = 1;
                    yield {
                        conversationId,
                        chunk: {
                            delta: finalContent.parts,
                            done: true
                        }
                    };
                }
                
                // 转换 XML 工具调用为 functionCall 格式（如果有）
                this.convertXMLToolCallsToFunctionCalls(finalContent);
                
                // 为没有 id 的 functionCall 添加唯一 id
                this.ensureFunctionCallIds(finalContent);
                
                // 保存 AI 响应到历史
                await this.conversationManager.addContent(conversationId, finalContent);
                
                // 检查是否有工具调用
                const functionCalls = this.extractFunctionCalls(finalContent);
                
                if (functionCalls.length === 0) {
                    // 没有工具调用，为模型消息创建存档点（如果配置了执行后）
                    const editModelMessageCheckpoints: CheckpointRecord[] = [];
                    if (this.checkpointManager && this.settingsManager?.shouldCreateAfterModelMessageCheckpoint()) {
                        const modelHistory = await this.conversationManager.getHistoryRef(conversationId);
                        const modelMessageIndex = modelHistory.length - 1;
                        
                        const checkpoint = await this.checkpointManager.createCheckpoint(
                            conversationId,
                            modelMessageIndex,
                            'model_message',
                            'after'
                        );
                        if (checkpoint) {
                            editModelMessageCheckpoints.push(checkpoint);
                        }
                    }
                    
                    // 结束循环
                    yield { conversationId, content: finalContent, checkpoints: editModelMessageCheckpoints };
                    return;
                }
                
                // 有工具调用，检查是否需要确认
                const editToolsNeedingConfirmation = this.getToolsNeedingConfirmation(functionCalls);
                
                if (editToolsNeedingConfirmation.length > 0) {
                    // 有工具需要确认，发送确认请求到前端
                    const pendingToolCalls: PendingToolCall[] = editToolsNeedingConfirmation.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }));
                    
                    yield {
                        conversationId,
                        pendingToolCalls,
                        content: finalContent,
                        awaitingConfirmation: true as const
                    };
                    
                    // 暂停执行，等待前端调用 handleToolConfirmation
                    return;
                }
                
                // 不需要确认，直接执行工具
                // 获取当前消息索引
                const editStreamHistory = await this.conversationManager.getHistoryRef(conversationId);
                const editStreamMessageIndex = editStreamHistory.length - 1;
                
                // 工具执行前先发送计时信息（让前端立即显示）
                yield {
                    conversationId,
                    content: finalContent,
                    toolsExecuting: true as const,
                    pendingToolCalls: functionCalls.map(call => ({
                        id: call.id,
                        name: call.name,
                        args: call.args
                    }))
                };
                
                const { responseParts, toolResults, checkpoints, multimodalAttachments: editMultimodalAttachments } = await this.executeFunctionCallsWithResults(
                    functionCalls,
                    conversationId,
                    editStreamMessageIndex,
                    config,
                    request.abortSignal
                );
                
                // 先将函数响应添加到历史（确保取消时也能保存）
                // 对于 XML/JSON 模式，如果有多模态附件，将其放在 parts 前面
                const editFunctionResponseParts = editMultimodalAttachments
                    ? [...editMultimodalAttachments, ...responseParts]
                    : responseParts;
                    
                await this.conversationManager.addContent(conversationId, {
                    role: 'user',
                    parts: editFunctionResponseParts,
                    isFunctionResponse: true
                });
                
                // 检查是否有工具被取消
                const editHasCancelled = toolResults.some(r => (r.result as any).cancelled);
                if (editHasCancelled) {
                    // 有工具被取消，发送最终的 toolIteration 后结束
                    yield {
                        conversationId,
                        content: finalContent,
                        toolIteration: true as const,
                        toolResults,
                        checkpoints
                    };
                    return;
                }
                
                // 发送 toolIteration 信号（包含检查点）
                yield {
                    conversationId,
                    content: finalContent,
                    toolIteration: true as const,
                    toolResults,
                    checkpoints
                };
                
                // 继续循环，让 AI 处理函数结果
            }
            
            // 达到最大迭代次数
            yield {
                conversationId,
                error: {
                    code: 'MAX_TOOL_ITERATIONS',
                    message: t('modules.api.chat.errors.maxToolIterations', { maxIterations: MAX_TOOL_CALL_ITERATIONS })
                }
            };
            
        } catch (error) {
            // 检查是否是用户取消错误
            if (error instanceof ChannelError && error.type === ErrorType.CANCELLED_ERROR) {
                // 用户取消，yield cancelled 消息
                yield {
                    conversationId: request.conversationId,
                    cancelled: true as const
                } as any;
                return;
            }
            
            yield {
                conversationId: request.conversationId,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 处理删除到指定消息的请求
     *
     * @param request 删除请求数据
     * @returns 删除响应数据
     */
    async handleDeleteToMessage(
        request: DeleteToMessageRequestData
    ): Promise<DeleteToMessageSuccessData | DeleteToMessageErrorData> {
        try {
            const { conversationId, targetIndex } = request;
            
            // 1. 确保对话存在
            await this.ensureConversation(conversationId);
            
            // 2. 中断之前未完成的 diff 等待
            const deleteDiffManager = getDiffManager();
            deleteDiffManager.markUserInterrupt();
            
            // 3. 删除关联的检查点（删除消息索引 >= targetIndex 的检查点）
            if (this.checkpointManager) {
                await this.checkpointManager.deleteCheckpointsFromIndex(conversationId, targetIndex);
            }
            
            // 4. 调用 ConversationManager 删除消息
            const deletedCount = await this.conversationManager.deleteToMessage(
                conversationId,
                targetIndex
            );
            
            // 4. 返回成功响应
            return {
                success: true,
                deletedCount
            };
            
        } catch (error) {
            // 错误处理
            return {
                success: false,
                error: this.formatError(error)
            };
        }
    }
    
    /**
     * 确保对话存在（不存在则创建）
     *
     * 由于 ConversationManager 现在无内存缓存，每次操作直接读写文件，
     * 只需调用 getHistory 即可触发自动创建逻辑（loadHistory 内部会处理）
     *
     * @param conversationId 对话 ID
     */
    private async ensureConversation(conversationId: string): Promise<void> {
        // getHistory 内部调用 loadHistory，如果对话不存在会自动创建
        await this.conversationManager.getHistory(conversationId);
    }
    
    /**
     * 检查并执行孤立的函数调用
     *
     * 如果历史最后一条 model 消息包含 functionCall 但没有对应的 functionResponse，
     * 则执行这些函数调用并将结果添加到历史
     *
     * @param conversationId 对话 ID
     * @returns 如果有孤立调用，返回执行结果；否则返回 null
     */
    private async checkAndExecuteOrphanedFunctionCalls(
        conversationId: string
    ): Promise<{
        functionCallContent: Content;
        toolResults: Array<{ id?: string; name: string; result: Record<string, unknown> }>;
    } | null> {
        const history = await this.conversationManager.getHistoryRef(conversationId);
        
        if (history.length === 0) {
            return null;
        }
        
        const lastMessage = history[history.length - 1];
        
        // 检查最后一条消息是否是 model 且包含 functionCall
        if (lastMessage.role !== 'model') {
            return null;
        }
        
        const functionCalls = this.extractFunctionCalls(lastMessage);
        if (functionCalls.length === 0) {
            return null;
        }
        
        // 检查是否有文本内容（如果有文本，说明函数调用已经完成）
        const hasTextContent = lastMessage.parts.some(p => p.text && !p.thought);
        if (hasTextContent) {
            return null;
        }
        
        // 执行这些函数调用
        // 获取当前消息索引
        const orphanedHistory = await this.conversationManager.getHistoryRef(conversationId);
        const orphanedMessageIndex = orphanedHistory.length - 1;
        
        const { responseParts, toolResults } = await this.executeFunctionCallsWithResults(
            functionCalls,
            conversationId,
            orphanedMessageIndex
        );
        
        // 将函数响应添加到历史（作为 user 消息，标记为函数响应）
        await this.conversationManager.addContent(conversationId, {
            role: 'user',
            parts: responseParts,
            isFunctionResponse: true
        });
        
        return {
            functionCallContent: lastMessage,
            toolResults
        };
    }
}