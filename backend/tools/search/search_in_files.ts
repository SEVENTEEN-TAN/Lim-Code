/**
 * 在文件中搜索内容工具
 *
 * 支持多工作区（Multi-root Workspaces）
 */

import * as vscode from 'vscode';
import type { Tool, ToolResult } from '../types';
import { getWorkspaceRoot, getAllWorkspaces, parseWorkspacePath, toRelativePath } from '../utils';
import { getGlobalSettingsManager } from '../../core/settingsContext';

/**
 * 默认排除模式
 */
const DEFAULT_EXCLUDE = '**/node_modules/**';

/**
 * 获取排除模式
 *
 * 从设置管理器获取用户配置的排除模式，如果未配置则使用默认值
 * 将多个模式合并为单个 glob 模式（用大括号语法）
 */
function getExcludePattern(): string {
    const settingsManager = getGlobalSettingsManager();
    if (settingsManager) {
        const config = settingsManager.getSearchInFilesConfig();
        if (config.excludePatterns && config.excludePatterns.length > 0) {
            // 多个模式用 {} 语法组合
            if (config.excludePatterns.length === 1) {
                return config.excludePatterns[0];
            }
            return `{${config.excludePatterns.join(',')}}`;
        }
    }
    return DEFAULT_EXCLUDE;
}

/**
 * 转义正则特殊字符
 */
function escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * 在单个目录中搜索
 */
async function searchInDirectory(
    searchRoot: vscode.Uri,
    filePattern: string,
    searchRegex: RegExp,
    maxResults: number,
    workspaceName: string | null,
    excludePattern: string
): Promise<Array<{
    file: string;
    workspace?: string;
    line: number;
    column: number;
    match: string;
    context: string;
}>> {
    const results: Array<{
        file: string;
        workspace?: string;
        line: number;
        column: number;
        match: string;
        context: string;
    }> = [];
    
    const pattern = new vscode.RelativePattern(searchRoot, filePattern);
    const files = await vscode.workspace.findFiles(pattern, excludePattern, 1000);
    
    for (const fileUri of files) {
        if (results.length >= maxResults) {
            break;
        }
        
        try {
            const content = await vscode.workspace.fs.readFile(fileUri);
            const text = new TextDecoder().decode(content);
            const lines = text.split('\n');
            
            for (let i = 0; i < lines.length; i++) {
                if (results.length >= maxResults) {
                    break;
                }
                
                const line = lines[i];
                let match;
                searchRegex.lastIndex = 0;
                
                while ((match = searchRegex.exec(line)) !== null) {
                    if (results.length >= maxResults) {
                        break;
                    }
                    
                    // 获取上下文（前后各1行）
                    const contextLines = [];
                    if (i > 0) {
                        contextLines.push(`${i}: ${lines[i - 1]}`);
                    }
                    contextLines.push(`${i + 1}: ${line}`);
                    if (i < lines.length - 1) {
                        contextLines.push(`${i + 2}: ${lines[i + 1]}`);
                    }
                    
                    // 使用支持多工作区的相对路径
                    const relativePath = toRelativePath(fileUri, workspaceName !== null);
                    
                    results.push({
                        file: relativePath,
                        workspace: workspaceName || undefined,
                        line: i + 1,
                        column: match.index + 1,
                        match: match[0],
                        context: contextLines.join('\n')
                    });
                }
            }
        } catch {
            // 跳过无法读取的文件
        }
    }
    
    return results;
}

/**
 * 创建搜索文件内容工具
 */
export function createSearchInFilesTool(): Tool {
    // 获取工作区信息用于描述
    const workspaces = getAllWorkspaces();
    const isMultiRoot = workspaces.length > 1;
    
    let pathDescription = 'Search directory (relative to workspace root), defaults to searching the entire workspace';
    if (isMultiRoot) {
        pathDescription = `Search directory, use "workspace_name/path" format or "." to search all workspaces. Available workspaces: ${workspaces.map(w => w.name).join(', ')}`;
    }
    
    return {
        declaration: {
            name: 'search_in_files',
            description: isMultiRoot
                ? `Search for content in multiple workspace files, supports regular expressions. Use "workspace_name/path" format to specify a workspace, or "." to search all. Available workspaces: ${workspaces.map(w => w.name).join(', ')}`
                : 'Search for content in workspace files, supports regular expressions. Returns matching files and context.',
            category: 'search',
            parameters: {
                type: 'object',
                properties: {
                    query: {
                        type: 'string',
                        description: 'Search keyword or regular expression'
                    },
                    path: {
                        type: 'string',
                        description: pathDescription,
                        default: '.'
                    },
                    pattern: {
                        type: 'string',
                        description: 'File matching pattern, e.g., "*.ts" or "**/*.js"',
                        default: '**/*'
                    },
                    isRegex: {
                        type: 'boolean',
                        description: 'Whether to use regular expressions',
                        default: false
                    },
                    maxResults: {
                        type: 'number',
                        description: 'Maximum number of results',
                        default: 100
                    }
                },
                required: ['query']
            }
        },
        handler: async (args): Promise<ToolResult> => {
            const query = args.query as string;
            const searchPath = (args.path as string) || '.';
            const filePattern = (args.pattern as string) || '**/*';
            const isRegex = (args.isRegex as boolean) || false;
            const maxResults = (args.maxResults as number) || 100;

            if (!query) {
                return { success: false, error: 'query is required' };
            }

            const workspaces = getAllWorkspaces();
            if (workspaces.length === 0) {
                return { success: false, error: 'No workspace folder open' };
            }

            try {
                const searchRegex = isRegex
                    ? new RegExp(query, 'gim')
                    : new RegExp(escapeRegex(query), 'gim');
                
                let allResults: Array<{
                    file: string;
                    workspace?: string;
                    line: number;
                    column: number;
                    match: string;
                    context: string;
                }> = [];
                
                // 获取排除模式
                const excludePattern = getExcludePattern();
                
                // 解析路径，确定搜索范围
                const { workspace: targetWorkspace, relativePath, isExplicit } = parseWorkspacePath(searchPath);
                
                if (isExplicit && targetWorkspace) {
                    // 显式指定了工作区，只搜索该工作区
                    const searchRoot = vscode.Uri.joinPath(targetWorkspace.uri, relativePath);
                    allResults = await searchInDirectory(
                        searchRoot,
                        filePattern,
                        searchRegex,
                        maxResults,
                        workspaces.length > 1 ? targetWorkspace.name : null,
                        excludePattern
                    );
                } else if (searchPath === '.' && workspaces.length > 1) {
                    // 搜索所有工作区
                    for (const ws of workspaces) {
                        if (allResults.length >= maxResults) break;
                        
                        const remaining = maxResults - allResults.length;
                        const wsResults = await searchInDirectory(
                            ws.uri,
                            filePattern,
                            searchRegex,
                            remaining,
                            ws.name,
                            excludePattern
                        );
                        allResults.push(...wsResults);
                    }
                } else {
                    // 单工作区或未指定，使用默认
                    const root = targetWorkspace?.uri || workspaces[0].uri;
                    const searchRoot = vscode.Uri.joinPath(root, relativePath);
                    allResults = await searchInDirectory(
                        searchRoot,
                        filePattern,
                        searchRegex,
                        maxResults,
                        workspaces.length > 1 ? (targetWorkspace?.name || workspaces[0].name) : null,
                        excludePattern
                    );
                }

                return {
                    success: true,
                    data: {
                        query,
                        results: allResults,
                        count: allResults.length,
                        truncated: allResults.length >= maxResults,
                        multiRoot: workspaces.length > 1
                    }
                };
            } catch (error) {
                return {
                    success: false,
                    error: `Search failed: ${error instanceof Error ? error.message : String(error)}`
                };
            }
        }
    };
}

/**
 * 注册搜索文件内容工具
 */
export function registerSearchInFiles(): Tool {
    return createSearchInFilesTool();
}