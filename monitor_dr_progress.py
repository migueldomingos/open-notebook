#!/usr/bin/env python3
"""
Deep Research Progress Monitor
Real-time tracking of workflow execution stages
"""

import requests
import time
import json
from datetime import datetime
from typing import Optional
import sys

API_URL = "http://localhost:8000"

class ProgressMonitor:
    def __init__(self, research_id: str):
        self.research_id = research_id
        self.start_time = time.time()
        self.last_stage = None
        
        # Define workflow stages in order
        self.workflow_stages = [
            ('plan_generation', 'Analyzing query'),
            ('search_questions', 'Generating questions'),
            ('knowledge_search', 'Knowledge search'),
            ('refinement', 'Refining results'),
            ('final_report', 'Final report'),
            ('completion', 'Complete'),
        ]
        
        self.stage_symbols = {
            'plan_generation': '📋',
            'search_questions': '❓',
            'knowledge_search': '🔍',
            'refinement': '✨',
            'final_report': '📄',
            'completion': '✅',
        }
        
    def get_progress(self) -> Optional[dict]:
        """Fetch current progress from API"""
        try:
            response = requests.get(
                f"{API_URL}/deep-research/progress/{self.research_id}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Error fetching progress: {e}")
        return None
    
    def get_result(self) -> Optional[dict]:
        """Fetch final result from API"""
        try:
            response = requests.get(
                f"{API_URL}/deep-research/full/{self.research_id}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching result: {e}")
        return None
    
    def print_status(self, progress: dict):
        """Pretty-print progress status with visual progress bar"""
        elapsed = time.time() - self.start_time
        current_stage = progress.get('current_stage', 'unknown')
        
        # Build progress bar
        bar_length = 50
        stage_index = next(
            (i for i, (stage, _) in enumerate(self.workflow_stages) if stage == current_stage),
            -1
        )
        
        if stage_index == -1:
            progress_percent = 0
            filled = 0
        else:
            progress_percent = ((stage_index + 1) / len(self.workflow_stages)) * 100
            filled = int((stage_index + 1) / len(self.workflow_stages) * bar_length)
        
        # Create visual progress bar
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Build stage indicators
        stage_indicators = []
        for i, (stage, label) in enumerate(self.workflow_stages):
            symbol = self.stage_symbols.get(stage, '○')
            if i < stage_index:
                # Completed stage
                stage_indicators.append(f"✅ {symbol}")
            elif i == stage_index:
                # Current stage
                stage_indicators.append(f"🔄 {symbol}")
            else:
                # Future stage
                stage_indicators.append(f"⭕ {symbol}")
        
        # Clear and print
        print("\n" + "="*70)
        print(f"📊 DEEP RESEARCH PROGRESS")
        print("="*70)
        
        # Progress bar
        print(f"\n{bar} {progress_percent:.0f}%")
        
        # Stage details
        print(f"\n📍 Stage Timeline:")
        for i, (stage, label) in enumerate(self.workflow_stages):
            indicator = stage_indicators[i]
            if i == stage_index:
                print(f"   {indicator} {label:<25} ← CURRENT")
            else:
                print(f"   {indicator} {label:<25}")
        
        # Live data
        print(f"\n⏱️  Elapsed: {elapsed:.1f}s")
        if current_stage:
            print(f"🎯 Current stage: {current_stage}")
        
        message = progress.get('message', 'N/A')
        if message:
            # Extract just the info part (after stage prefix)
            msg_clean = message.split('] ', 1)[-1] if '] ' in message else message
            print(f"📝 {msg_clean}")
        
        # Additional metrics
        if 'iteration' in progress:
            iteration = progress['iteration']
            print(f"🔄 Iteration: {iteration}")
        
        if 'search_progress' in progress:
            search = progress['search_progress']
            found = search.get('found_results', 0)
            if found > 0:
                print(f"🔍 Documents found: {found}")
        
        if 'llm_calls_made' in progress:
            llm_calls = progress['llm_calls_made']
            if llm_calls > 0:
                print(f"🤖 LLM calls made: {llm_calls}")
        
        if 'confidence_score' in progress and progress['confidence_score'] is not None:
            confidence = progress['confidence_score']
            conf_bar = '█' * int(confidence * 20) + '░' * (20 - int(confidence * 20))
            print(f"🏆 Confidence: [{conf_bar}] {confidence:.1%}")
        
        print("="*70)
    
    def monitor(self, poll_interval: int = 2, timeout: int = 600):
        """Monitor progress in real-time"""
        print(f"\n🔬 Monitoring Deep Research: {self.research_id}")
        print(f"⏱️  Timeout: {timeout}s")
        
        last_message = None
        
        while time.time() - self.start_time < timeout:
            progress = self.get_progress()
            
            if not progress:
                print("⏳ Waiting for research to start...")
                time.sleep(poll_interval)
                continue
            
            # Print update if message changed
            current_message = progress.get('message', '')
            if current_message != last_message:
                self.print_status(progress)
                last_message = current_message
            
            # Check if completed
            if progress.get('status') == 'completed':
                print("\n✅ Research completed!")
                result = self.get_result()
                if result:
                    print(f"\n📊 Results:")
                    print(f"   - Synthesis length: {len(result.get('synthesis', ''))} chars")
                    print(f"   - Confidence: {result.get('confidence_score', 'N/A')}")
                    print(f"   - Total iterations: {result.get('iterations', 'N/A')}")
                    print(f"   - Sources found: {result.get('sources_count', 0)}")
                return True
            
            if progress.get('status') == 'failed':
                print(f"\n❌ Research failed!")
                print(f"   Error: {progress.get('error', 'Unknown error')}")
                return False
            
            time.sleep(poll_interval)
        
        print(f"\n⏱️  Timeout reached ({timeout}s)")
        progress = self.get_progress()
        if progress:
            self.print_status(progress)
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_dr_progress.py <research_id>")
        print("\nExample: python monitor_dr_progress.py 550e8400-e29b-41d4-a716-446655440000")
        sys.exit(1)
    
    research_id = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    
    monitor = ProgressMonitor(research_id)
    success = monitor.monitor(timeout=timeout)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
