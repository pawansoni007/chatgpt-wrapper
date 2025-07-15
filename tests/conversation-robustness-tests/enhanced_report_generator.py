#!/usr/bin/env python3
"""
Enhanced HTML Report Generator for Conversation Tests
==================================================

A comprehensive HTML report generator that provides:
- Interactive charts and visualizations
- Detailed conversation analysis
- Performance metrics dashboard
- Responsive design with modern UI
- Filtering and search capabilities
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import statistics

class EnhancedHTMLReportGenerator:
    """Generate enhanced HTML reports with interactive visualizations"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results_data = self.load_results()
        self.summary_metrics = self.calculate_summary_metrics()
        
    def load_results(self) -> Dict:
        """Load JSON test results"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded test results from: {self.results_file}")
            return data
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in results file: {e}")
            sys.exit(1)
    
    def calculate_summary_metrics(self) -> Dict:
        """Calculate comprehensive summary metrics"""
        test_suite_summary = self.results_data.get('test_suite_summary', {})
        individual_results = self.results_data.get('individual_test_results', {})
        
        # Basic metrics
        total_tests = test_suite_summary.get('total_tests', 0)
        successful_tests = test_suite_summary.get('successful_tests', 0)
        success_rate = (successful_tests / max(total_tests, 1)) * 100
        
        # Detailed metrics from individual test results
        response_times = []
        token_usage = []
        turn_counts = []
        intent_accuracies = []
        context_scores = []
        
        for test_name, test_data in individual_results.items():
            exec_summary = test_data.get('execution_summary', {})
            conv_analysis = test_data.get('conversation_analysis', {})
            messages = test_data.get('message_responses', [])
            
            # Response time metrics
            if 'avg_response_time' in exec_summary:
                response_times.append(exec_summary['avg_response_time'])
            
            # Turn count metrics
            if 'total_turns' in exec_summary:
                turn_counts.append(exec_summary['total_turns'])
            
            # Token usage from messages
            for msg in messages:
                if 'tokens_used' in msg:
                    token_usage.append(msg['tokens_used'])
            
            # Analysis scores
            if 'intent_accuracy_score' in conv_analysis:
                intent_accuracies.append(conv_analysis['intent_accuracy_score'])
            if 'context_quality_score' in conv_analysis:
                context_scores.append(conv_analysis['context_quality_score'])
        
        # Calculate statistics
        return {
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'total_duration': test_suite_summary.get('total_duration', 0),
            'total_turns': test_suite_summary.get('total_conversation_turns', 0),
            'response_times': {
                'avg': statistics.mean(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0
            },
            'token_usage': {
                'avg': statistics.mean(token_usage) if token_usage else 0,
                'min': min(token_usage) if token_usage else 0,
                'max': max(token_usage) if token_usage else 0,
                'total': sum(token_usage) if token_usage else 0
            },
            'turn_counts': {
                'avg': statistics.mean(turn_counts) if turn_counts else 0,
                'min': min(turn_counts) if turn_counts else 0,
                'max': max(turn_counts) if turn_counts else 0
            },
            'intent_accuracy': {
                'avg': statistics.mean(intent_accuracies) * 100 if intent_accuracies else 0,
                'min': min(intent_accuracies) * 100 if intent_accuracies else 0,
                'max': max(intent_accuracies) * 100 if intent_accuracies else 0
            },
            'context_quality': {
                'avg': statistics.mean(context_scores) * 100 if context_scores else 0,
                'min': min(context_scores) * 100 if context_scores else 0,
                'max': max(context_scores) * 100 if context_scores else 0
            }
        }
    
    def generate_chart_data(self) -> Dict:
        """Generate data for charts and visualizations"""
        individual_results = self.results_data.get('individual_test_results', {})
        
        test_performance = []
        response_time_data = []
        token_usage_data = []
        
        for test_name, test_data in individual_results.items():
            exec_summary = test_data.get('execution_summary', {})
            conv_analysis = test_data.get('conversation_analysis', {})
            messages = test_data.get('message_responses', [])
            
            # Test performance data
            test_performance.append({
                'name': test_name.replace('_', ' ').title(),
                'success_rate': (exec_summary.get('successful_turns', 0) / 
                               max(exec_summary.get('total_turns', 1), 1)) * 100,
                'avg_response_time': exec_summary.get('avg_response_time', 0),
                'total_turns': exec_summary.get('total_turns', 0),
                'intent_accuracy': conv_analysis.get('intent_accuracy_score', 0) * 100,
                'context_quality': conv_analysis.get('context_quality_score', 0) * 100
            })
            
            # Response time over turns
            turn_times = []
            for i, msg in enumerate(messages, 1):
                turn_times.append({
                    'turn': i,
                    'response_time': msg.get('response_time', 0),
                    'test_name': test_name
                })
            response_time_data.extend(turn_times)
            
            # Token usage over turns
            turn_tokens = []
            for i, msg in enumerate(messages, 1):
                turn_tokens.append({
                    'turn': i,
                    'tokens': msg.get('tokens_used', 0),
                    'test_name': test_name
                })
            token_usage_data.extend(turn_tokens)
        
        return {
            'test_performance': test_performance,
            'response_time_data': response_time_data,
            'token_usage_data': token_usage_data
        }
    
    def generate_html_report(self, output_file: str = None) -> str:
        """Generate comprehensive HTML report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"enhanced_conversation_report_{timestamp}.html"
        
        chart_data = self.generate_chart_data()
        individual_results = self.results_data.get('individual_test_results', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Conversation Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 3em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            color: #666;
            margin-bottom: 20px;
        }}
        
        .nav-tabs {{
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .nav-tab {{
            padding: 15px 30px;
            margin: 0 5px;
            background: transparent;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            color: #666;
        }}
        
        .nav-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .tab-content {{
            display: none;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            text-align: center;
            border: 1px solid #f0f0f0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }}
        
        .metric-card h3 {{
            color: #667eea;
            font-size: 1.1em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: 700;
            margin: 15px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-details {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .test-grid {{
            display: grid;
            gap: 25px;
            margin-top: 30px;
        }}
        
        .test-card {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        
        .test-card:hover {{
            transform: translateY(-3px);
        }}
        
        .test-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .test-title {{
            font-size: 1.3em;
            font-weight: 600;
        }}
        
        .test-status {{
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .test-content {{
            padding: 0;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        
        .test-content.expanded {{
            max-height: none;
        }}
        
        .test-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            padding: 25px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .conversation-section {{
            padding: 25px;
            max-height: 600px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #667eea #f1f1f1;
        }}
        
        .conversation-section::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .conversation-section::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        
        .conversation-section::-webkit-scrollbar-thumb {{
            background: #667eea;
            border-radius: 4px;
        }}
        
        .conversation-section::-webkit-scrollbar-thumb:hover {{
            background: #5a6fd8;
        }}
        
        .conversation-turn {{
            margin-bottom: 25px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid #f0f0f0;
        }}
        
        .message {{
            padding: 20px;
            position: relative;
        }}
        
        .user-message {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
        }}
        
        .assistant-message {{
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-left: 4px solid #9c27b0;
        }}
        
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .message-role {{
            font-weight: 600;
            color: #333;
        }}
        
        .message-meta {{
            font-size: 0.8em;
            color: #666;
        }}
        
        .message-content {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.05);
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.95em;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            word-break: break-word;
            scrollbar-width: thin;
            scrollbar-color: #ccc #f9f9f9;
        }}
        
        .message-content::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .message-content::-webkit-scrollbar-track {{
            background: #f9f9f9;
            border-radius: 3px;
        }}
        
        .message-content::-webkit-scrollbar-thumb {{
            background: #ccc;
            border-radius: 3px;
        }}
        
        .message-content::-webkit-scrollbar-thumb:hover {{
            background: #999;
        }}
        
        .search-filter {{
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .search-input {{
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .filter-select {{
            padding: 12px 20px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            background: white;
            cursor: pointer;
        }}
        
        .expand-all-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        
        .expand-all-btn:hover {{
            transform: translateY(-2px);
        }}
        
        .footer {{
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 50px;
            padding: 30px;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .nav-tabs {{
                flex-direction: column;
                gap: 10px;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .search-filter {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Enhanced Test Report</h1>
            <div class="subtitle">Comprehensive Analysis of Conversation Performance</div>
            <div style="margin-top: 15px; font-size: 0.9em; color: #888;">
                Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">📈 Overview</button>
            <button class="nav-tab" onclick="showTab('charts')">📊 Charts</button>
            <button class="nav-tab" onclick="showTab('tests')">🧪 Test Details</button>
            <button class="nav-tab" onclick="showTab('conversations')">💬 Conversations</button>
        </div>
        
        <div id="overview" class="tab-content active">
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Success Rate</h3>
                    <div class="metric-value">{self.summary_metrics['success_rate']:.1f}%</div>
                    <div class="metric-details">
                        {self.summary_metrics['successful_tests']}/{self.summary_metrics['total_tests']} tests passed
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Total Conversations</h3>
                    <div class="metric-value">{self.summary_metrics['total_turns']}</div>
                    <div class="metric-details">
                        Across all test scenarios
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Avg Response Time</h3>
                    <div class="metric-value">{self.summary_metrics['response_times']['avg']:.2f}s</div>
                    <div class="metric-details">
                        Range: {self.summary_metrics['response_times']['min']:.2f}s - {self.summary_metrics['response_times']['max']:.2f}s
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Token Usage</h3>
                    <div class="metric-value">{self.summary_metrics['token_usage']['total']:,}</div>
                    <div class="metric-details">
                        Avg: {self.summary_metrics['token_usage']['avg']:.0f} per message
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Intent Accuracy</h3>
                    <div class="metric-value">{self.summary_metrics['intent_accuracy']['avg']:.1f}%</div>
                    <div class="metric-details">
                        Classification performance
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Context Quality</h3>
                    <div class="metric-value">{self.summary_metrics['context_quality']['avg']:.1f}%</div>
                    <div class="metric-details">
                        Conversation coherence
                    </div>
                </div>
            </div>
        </div>
        
        <div id="charts" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Test Performance Comparison</div>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Response Time Distribution</div>
                <canvas id="responseTimeChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Token Usage Over Time</div>
                <canvas id="tokenChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div id="tests" class="tab-content">
            <div class="search-filter">
                <input type="text" class="search-input" placeholder="Search tests..." onkeyup="filterTests()">
                <select class="filter-select" onchange="filterTests()">
                    <option value="">All Tests</option>
                    <option value="passed">Passed Only</option>
                    <option value="failed">Failed Only</option>
                </select>
                <button class="expand-all-btn" onclick="toggleAllTests()">Expand All</button>
            </div>
            
            <div class="test-grid" id="testGrid">
"""

        # Add test details
        for test_name, test_data in individual_results.items():
            exec_summary = test_data.get('execution_summary', {})
            conv_analysis = test_data.get('conversation_analysis', {})
            messages = test_data.get('message_responses', [])
            errors = test_data.get('errors', [])
            
            status = "passed" if exec_summary.get('errors', 0) == 0 else "failed"
            success_rate = (exec_summary.get('successful_turns', 0) / 
                           max(exec_summary.get('total_turns', 1), 1)) * 100
            
            html_content += f"""
                <div class="test-card" data-test-name="{test_name}" data-status="{status}">
                    <div class="test-header" onclick="toggleTest('{test_name}')">
                        <div class="test-title">{test_name.replace('_', ' ').title()}</div>
                        <div class="test-status">{'✅ Passed' if status == 'passed' else '❌ Failed'}</div>
                    </div>
                    <div class="test-content" id="test-{test_name}">
                        <div class="test-metrics">
                            <div class="metric">
                                <div class="metric-label">Success Rate</div>
                                <div class="metric-value">{success_rate:.1f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Total Turns</div>
                                <div class="metric-value">{exec_summary.get('total_turns', 0)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Avg Response</div>
                                <div class="metric-value">{exec_summary.get('avg_response_time', 0):.2f}s</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Duration</div>
                                <div class="metric-value">{exec_summary.get('total_duration', 0):.1f}s</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Intent Accuracy</div>
                                <div class="metric-value">{conv_analysis.get('intent_accuracy_score', 0) * 100:.1f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Context Quality</div>
                                <div class="metric-value">{conv_analysis.get('context_quality_score', 0) * 100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                </div>
"""

        html_content += """
            </div>
        </div>
        
        <div id="conversations" class="tab-content">
            <div class="search-filter">
                <input type="text" class="search-input" placeholder="Search conversations..." onkeyup="filterConversations()">
                <select class="filter-select" onchange="filterConversations()">
                    <option value="">All Conversations</option>
"""

        # Add conversation filter options
        for test_name in individual_results.keys():
            html_content += f'<option value="{test_name}">{test_name.replace("_", " ").title()}</option>'

        html_content += """
                </select>
            </div>
            
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 4px solid #2196f3;">
                <h3 style="margin: 0 0 10px 0; color: #1976d2; font-size: 1.1em;">💡 Viewing Conversations</h3>
                <p style="margin: 0; color: #666; font-size: 0.95em;">Each conversation is automatically expanded below. You can scroll within each conversation section to view all messages. Long messages have their own scroll areas for easier reading.</p>
            </div>
            
            <div id="conversationGrid">
"""

        # Add conversation details
        for test_name, test_data in individual_results.items():
            messages = test_data.get('message_responses', [])
            conv_details = test_data.get('conversation_details', {})
            
            if messages:
                html_content += f"""
                <div class="test-card conversation-card" data-test-name="{test_name}">
                    <div class="test-header" onclick="toggleConversation('{test_name}')">
                        <div class="test-title">{test_name.replace('_', ' ').title()}</div>
                        <div class="test-status">{len(messages)} turns</div>
                    </div>
                    <div class="test-content expanded" id="conv-{test_name}">
                        <div class="conversation-section">
"""

                for i, msg in enumerate(messages, 1):
                    user_msg = msg.get('user_message', '')
                    assistant_msg = msg.get('assistant_response', '')
                    response_time = msg.get('response_time', 0)
                    tokens = msg.get('tokens_used', 0)
                    expected_intent = msg.get('expected_intent', '')
                    
                    html_content += f"""
                            <div class="conversation-turn">
                                <div class="message user-message">
                                    <div class="message-header">
                                        <span class="message-role">👤 User (Turn {i})</span>
                                        <span class="message-meta">Expected: {expected_intent}</span>
                                    </div>
                                    <div class="message-content">{user_msg}</div>
                                </div>
                                
                                <div class="message assistant-message">
                                    <div class="message-header">
                                        <span class="message-role">🤖 Assistant</span>
                                        <span class="message-meta">
                                            {response_time:.2f}s • {tokens} tokens
                                        </span>
                                    </div>
                                    <div class="message-content">{assistant_msg}</div>
                                </div>
                            </div>
"""

                html_content += """
                        </div>
                    </div>
                </div>
"""

        # Add JavaScript and closing HTML
        html_content += f"""
            </div>
        </div>
        
        <div class="footer">
            <p>Enhanced Conversation Test Report • Generated with ❤️ on {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
    </div>

    <script>
        // Tab functionality
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.nav-tab').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'charts') {{
                setTimeout(initCharts, 100);
            }}
        }}
        
        // Test toggle functionality
        function toggleTest(testName) {{
            const content = document.getElementById(`test-${{testName}}`);
            content.classList.toggle('expanded');
        }}
        
        function toggleConversation(testName) {{
            const content = document.getElementById(`conv-${{testName}}`);
            content.classList.toggle('expanded');
        }}
        
        let allTestsExpanded = false;
        function toggleAllTests() {{
            const contents = document.querySelectorAll('.test-content');
            const btn = document.querySelector('.expand-all-btn');
            
            contents.forEach(content => {{
                if (allTestsExpanded) {{
                    content.classList.remove('expanded');
                }} else {{
                    content.classList.add('expanded');
                }}
            }});
            
            allTestsExpanded = !allTestsExpanded;
            btn.textContent = allTestsExpanded ? 'Collapse All' : 'Expand All';
        }}
        
        // Filter functionality
        function filterTests() {{
            const searchTerm = document.querySelector('.search-input').value.toLowerCase();
            const statusFilter = document.querySelector('.filter-select').value;
            const testCards = document.querySelectorAll('.test-card');
            
            testCards.forEach(card => {{
                const testName = card.dataset.testName.toLowerCase();
                const status = card.dataset.status;
                
                const matchesSearch = testName.includes(searchTerm);
                const matchesStatus = !statusFilter || status === statusFilter;
                
                if (matchesSearch && matchesStatus) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
        
        function filterConversations() {{
            const searchTerm = document.querySelector('#conversations .search-input').value.toLowerCase();
            const testFilter = document.querySelector('#conversations .filter-select').value;
            const conversationCards = document.querySelectorAll('.conversation-card');
            
            conversationCards.forEach(card => {{
                const testName = card.dataset.testName.toLowerCase();
                
                const matchesSearch = testName.includes(searchTerm);
                const matchesTest = !testFilter || testName.includes(testFilter);
                
                if (matchesSearch && matchesTest) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
        
        // Chart initialization
        function initCharts() {{
            const chartData = {json.dumps(chart_data)};
            
            // Performance Chart
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            new Chart(performanceCtx, {{
                type: 'bar',
                data: {{
                    labels: chartData.test_performance.map(t => t.name),
                    datasets: [{{
                        label: 'Success Rate (%)',
                        data: chartData.test_performance.map(t => t.success_rate),
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }}, {{
                        label: 'Intent Accuracy (%)',
                        data: chartData.test_performance.map(t => t.intent_accuracy),
                        backgroundColor: 'rgba(118, 75, 162, 0.8)',
                        borderColor: 'rgba(118, 75, 162, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }}
                }}
            }});
            
            // Response Time Chart
            const responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
            const responseTimeData = {{}};
            chartData.response_time_data.forEach(item => {{
                if (!responseTimeData[item.test_name]) {{
                    responseTimeData[item.test_name] = [];
                }}
                responseTimeData[item.test_name].push(item.response_time);
            }});
            
            new Chart(responseTimeCtx, {{
                type: 'line',
                data: {{
                    labels: Array.from({{length: Math.max(...Object.values(responseTimeData).map(arr => arr.length))}}, (_, i) => i + 1),
                    datasets: Object.keys(responseTimeData).map((testName, index) => ({{
                        label: testName.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
                        data: responseTimeData[testName],
                        borderColor: `hsl(${{index * 60}}, 70%, 50%)`,
                        backgroundColor: `hsla(${{index * 60}}, 70%, 50%, 0.1)`,
                        tension: 0.4
                    }}))
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Response Time (seconds)'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Turn Number'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Token Usage Chart
            const tokenCtx = document.getElementById('tokenChart').getContext('2d');
            const tokenData = {{}};
            chartData.token_usage_data.forEach(item => {{
                if (!tokenData[item.test_name]) {{
                    tokenData[item.test_name] = [];
                }}
                tokenData[item.test_name].push(item.tokens);
            }});
            
            new Chart(tokenCtx, {{
                type: 'bar',
                data: {{
                    labels: Array.from({{length: Math.max(...Object.values(tokenData).map(arr => arr.length))}}, (_, i) => i + 1),
                    datasets: Object.keys(tokenData).map((testName, index) => ({{
                        label: testName.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
                        data: tokenData[testName],
                        backgroundColor: `hsla(${{index * 60}}, 70%, 50%, 0.8)`,
                        borderColor: `hsl(${{index * 60}}, 70%, 50%)`,
                        borderWidth: 1
                    }}))
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Tokens Used'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Turn Number'
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            // Any initialization code here
        }});
    </script>
</body>
</html>
"""

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced HTML report generated: {output_file}")
        return output_file

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Generate enhanced HTML conversation test report")
    parser.add_argument('results_file', help='JSON results file from conversation tests')
    parser.add_argument('--output', help='Output HTML file name (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        sys.exit(1)
    
    try:
        generator = EnhancedHTMLReportGenerator(args.results_file)
        html_file = generator.generate_html_report(args.output)
        
        print(f"\n🎉 Enhanced report generation complete!")
        print(f"📄 HTML Report: {html_file}")
        print(f"\n💡 Open {html_file} in your browser to view the interactive report!")
        print(f"✨ Features: Interactive charts, conversation analysis, search & filter, responsive design")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()