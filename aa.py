claude

┌─────────────────────────────────┐
│   Legacy PDF    │   Modern PDF  │
└────────┬────────────────┬───────┘
         │                │
         ▼                ▼
    ┌─────────────────────────┐
    │ Component 1: Docling    │
    │ Extract + structure     │
    │ Cost: Free              │
    │ Time: 2-5s              │
    └────────┬────────────────┘
             │
    ┌────────▼────────────────┐
    │ Component 2: Index      │
    │ Create lightweight map  │
    │ Cost: Free              │
    │ Time: <1s               │
    └────────┬────────────────┘
             │
    ┌────────▼────────────────┐
    │ Component 3: Diff       │
    │ Find changed sections   │
    │ Cost: Free              │
    │ Time: <1s               │
    │ Result: 12 sections     │
    └────────┬────────────────┘
             │
    ┌────────▼────────────────┐
    │ Component 4: Retrieve   │
    │ Load only changed       │
    │ Cost: Memory-efficient  │
    │ Time: <1s               │
    │ Data: 15k tokens        │
    └────────┬────────────────┘
             │
    ┌────────▼────────────────┐
    │ Component 5: Qwen LLM   │
    │ Semantic analysis       │
    │ Cost: Free (local)      │
    │ Time: 12 × 10s = 120s   │
    │ Output: Structured JSON │
    └────────┬────────────────┘
             │
    ┌────────▼────────────────┐
    │ Component 6: Report     │
    │ Aggregate + format      │
    │ Cost: Free              │
    │ Time: 1s                │
    │ Output: Report + CSV    │
    └────────┬────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │  Executive Report        │
    │  Risk Score: HIGH        │
    │  Compliance Checklist    │
    │  Regulatory Mapping      │
    └──────────────────────────┘

Total: ~130 seconds (2 min) end-to-end


# ============================================================================
# JUPYTER NOTEBOOK: Policy Comparison Pipeline
# ============================================================================
# Copy all code below into your Jupyter notebook cells
# ============================================================================

# Cell 1: Install dependencies (run once)
# !pip install docling transformers torch
# For ROCm: !pip install torch-rocm

# ============================================================================
# CELL 1: IMPORTS & SETUP
# ============================================================================

from docling.document_converter import DocumentConverter
import re
from typing import Dict, List, Tuple
from collections import Counter
import json
from datetime import datetime
import time

print("✓ All imports successful")

# ============================================================================
# CELL 2: COMPONENT 1 - PDF PARSER (DOCLING)
# ============================================================================

def parse_pdf(pdf_path: str) -> str:
    """Parse PDF using Docling and return markdown"""
    print(f"  Parsing {pdf_path}...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    print(f"  ✓ Extracted {len(markdown)} characters")
    return markdown

def extract_sections(markdown_text: str) -> Dict[str, Dict]:
    """Extract sections from markdown"""
    sections = {}
    lines = markdown_text.split('\n')
    current_section = None
    current_content = []
    section_id = 0
    
    for line_num, line in enumerate(lines):
        heading_match = re.match(r'^(#{1,6})\s+(.+?)$', line.strip())
        
        if heading_match:
            # Save previous section
            if current_section:
                sections[current_section['title']] = {
                    'id': section_id,
                    'level': current_section['level'],
                    'title': current_section['title'],
                    'start_line': current_section['start_line'],
                    'end_line': line_num,
                    'content': '\n'.join(current_content),
                    'line_count': line_num - current_section['start_line']
                }
                section_id += 1
            
            # Create new section
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_section = {
                'level': heading_level,
                'title': heading_text,
                'start_line': line_num,
            }
            current_content = []
        
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section['title']] = {
            'id': section_id,
            'level': current_section['level'],
            'title': current_section['title'],
            'start_line': current_section['start_line'],
            'end_line': len(lines),
            'content': '\n'.join(current_content),
            'line_count': len(lines) - current_section['start_line']
        }
    
    return sections

def estimate_tokens(text: str) -> int:
    """Estimate tokens: ~1 token per 4 chars"""
    return len(text) // 4

def print_structure(sections: Dict, name: str = "Document") -> int:
    """Print document structure"""
    print(f"\n{'='*80}")
    print(f"Document Structure: {name}")
    print(f"Total Sections: {len(sections)}")
    print(f"{'='*80}\n")
    
    total_tokens = 0
    for title, info in sorted(sections.items(), key=lambda x: x[1]['id']):
        tokens = estimate_tokens(info['content'])
        level = info['level']
        indent = "  " * (level - 1)
        print(f"{indent}[{info['id']:2d}] {title}")
        print(f"{indent}    Tokens: {tokens:5d} | Lines: {info['line_count']:4d}")
        total_tokens += tokens
    
    print(f"\n{'='*80}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"{'='*80}\n")
    
    return total_tokens

# Test Component 1
print("Component 1: PDF Parser - READY")

# ============================================================================
# CELL 3: COMPONENT 2 - PROXY INDEX BUILDER
# ============================================================================

def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """Extract important keywords from section"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'shall', 'can', 'that', 'this', 'these', 'those', 'it', 'its',
        'we', 'you', 'they', 'he', 'she', 'which', 'who', 'what', 'when', 'where'
    }
    
    words = re.findall(r'\b[a-z0-9_\-]+\b', text.lower())
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    counter = Counter(words)
    keywords = [word for word, _ in counter.most_common(top_n)]
    
    return keywords

def build_index(sections: Dict) -> Dict:
    """Build lightweight proxy index from sections"""
    index = {}
    
    for section_title, section_data in sections.items():
        content = section_data['content']
        keywords = extract_keywords(content)
        
        index[section_title] = {
            'id': section_data['id'],
            'title': section_title,
            'level': section_data['level'],
            'start_line': section_data['start_line'],
            'end_line': section_data['end_line'],
            'content_length': len(content),
            'token_estimate': estimate_tokens(content),
            'line_count': section_data['line_count'],
            'keywords': keywords,
            'hash': hash(content)
        }
    
    return index

def compare_indexes(legacy_index: Dict, modern_index: Dict) -> Dict:
    """Compare two indexes to identify changes"""
    legacy_sections = set(legacy_index.keys())
    modern_sections = set(modern_index.keys())
    
    comparison = {
        'removed': sorted(list(legacy_sections - modern_sections)),
        'added': sorted(list(modern_sections - legacy_sections)),
        'modified': sorted(list(legacy_sections & modern_sections)),
        'total_legacy_sections': len(legacy_sections),
        'total_modern_sections': len(modern_sections),
        'legacy_total_tokens': sum(s['token_estimate'] for s in legacy_index.values()),
        'modern_total_tokens': sum(s['token_estimate'] for s in modern_index.values()),
    }
    
    return comparison

def print_comparison(comparison: Dict):
    """Print comparison results"""
    print(f"\n{'='*80}")
    print("DIFF DETECTION RESULTS (Structure Only - No LLM)")
    print(f"{'='*80}\n")
    
    print(f"REMOVED: {len(comparison['removed'])} sections")
    for section in comparison['removed'][:5]:
        print(f"  • {section}")
    if len(comparison['removed']) > 5:
        print(f"  ... and {len(comparison['removed']) - 5} more")
    
    print(f"\nADDED: {len(comparison['added'])} sections")
    for section in comparison['added'][:5]:
        print(f"  • {section}")
    if len(comparison['added']) > 5:
        print(f"  ... and {len(comparison['added']) - 5} more")
    
    print(f"\nMODIFIED: {len(comparison['modified'])} sections")
    for section in comparison['modified'][:5]:
        print(f"  • {section}")
    if len(comparison['modified']) > 5:
        print(f"  ... and {len(comparison['modified']) - 5} more")
    
    print(f"\nLegacy total tokens: {comparison['legacy_total_tokens']:,}")
    print(f"Modern total tokens: {comparison['modern_total_tokens']:,}")
    print(f"Token difference: {comparison['modern_total_tokens'] - comparison['legacy_total_tokens']:+,}")
    print(f"\n{'='*80}\n")

print("Component 2: Proxy Index Builder - READY")


# ============================================================================
# CELL 4: COMPONENT 3 - CONTENT RETRIEVAL
# ============================================================================

def get_section_pair(legacy_sections: Dict, modern_sections: Dict, section_name: str) -> Tuple[str, str]:
    """Get section content pair (legacy, modern)"""
    legacy_content = None
    modern_content = None
    
    if section_name in legacy_sections:
        legacy_content = legacy_sections[section_name]['content']
    
    if section_name in modern_sections:
        modern_content = modern_sections[section_name]['content']
    
    return legacy_content, modern_content

def prepare_pairs(legacy_sections: Dict, modern_sections: Dict, comparison: Dict) -> Dict:
    """Prepare section pairs for analysis"""
    pairs = {
        'modified': [],
        'removed': [],
        'added': []
    }
    
    # Modified
    for section_name in comparison['modified']:
        legacy_content, modern_content = get_section_pair(legacy_sections, modern_sections, section_name)
        if legacy_content and modern_content:
            pairs['modified'].append({
                'section_name': section_name,
                'legacy_content': legacy_content,
                'modern_content': modern_content,
                'legacy_tokens': estimate_tokens(legacy_content),
                'modern_tokens': estimate_tokens(modern_content)
            })
    
    # Removed
    for section_name in comparison['removed']:
        legacy_content, _ = get_section_pair(legacy_sections, modern_sections, section_name)
        if legacy_content:
            pairs['removed'].append({
                'section_name': section_name,
                'legacy_content': legacy_content,
                'legacy_tokens': estimate_tokens(legacy_content)
            })
    
    # Added
    for section_name in comparison['added']:
        _, modern_content = get_section_pair(legacy_sections, modern_sections, section_name)
        if modern_content:
            pairs['added'].append({
                'section_name': section_name,
                'modern_content': modern_content,
                'modern_tokens': estimate_tokens(modern_content)
            })
    
    return pairs

def truncate_section(text: str, max_chars: int = 2000) -> str:
    """Truncate section if too long"""
    if len(text) <= max_chars:
        return text
    
    chars_keep_start = int(max_chars * 0.7)
    chars_keep_end = int(max_chars * 0.3)
    
    truncated = text[:chars_keep_start] + "\n\n[... TRUNCATED ...]\n\n" + text[-chars_keep_end:]
    return truncated

def print_pairs_summary(pairs: Dict):
    """Print summary of retrieved pairs"""
    print(f"\n{'='*80}")
    print("CONTENT RETRIEVAL SUMMARY")
    print(f"{'='*80}\n")
    
    if pairs['modified']:
        total_tokens = sum(p['legacy_tokens'] + p['modern_tokens'] for p in pairs['modified'])
        print(f"MODIFIED ({len(pairs['modified'])} sections, ~{total_tokens:,} tokens):")
        for pair in pairs['modified'][:5]:
            print(f"  • {pair['section_name']}: {pair['legacy_tokens']:,} → {pair['modern_tokens']:,} tokens")
        if len(pairs['modified']) > 5:
            print(f"  ... and {len(pairs['modified']) - 5} more")
    
    if pairs['removed']:
        total_tokens = sum(p['legacy_tokens'] for p in pairs['removed'])
        print(f"\nREMOVED ({len(pairs['removed'])} sections, ~{total_tokens:,} tokens):")
        for pair in pairs['removed'][:5]:
            print(f"  • {pair['section_name']} ({pair['legacy_tokens']:,} tokens)")
        if len(pairs['removed']) > 5:
            print(f"  ... and {len(pairs['removed']) - 5} more")
    
    if pairs['added']:
        total_tokens = sum(p['modern_tokens'] for p in pairs['added'])
        print(f"\nADDED ({len(pairs['added'])} sections, ~{total_tokens:,} tokens):")
        for pair in pairs['added'][:5]:
            print(f"  • {pair['section_name']} ({pair['modern_tokens']:,} tokens)")
        if len(pairs['added']) > 5:
            print(f"  ... and {len(pairs['added']) - 5} more")
    
    print(f"\n{'='*80}\n")

print("Component 3: Content Retrieval - READY")


# ============================================================================
# CELL 5: COMPONENT 4 - QWEN LLM ANALYZER
# ============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """Initialize Qwen LLM"""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"✓ Model loaded on {next(self.model.parameters()).device}")
    
    def analyze_section_pair(self, section_name: str, legacy_content: str, modern_content: str) -> Dict:
        """Analyze semantic differences between sections"""
        legacy_content = truncate_section(legacy_content, 2000)
        modern_content = truncate_section(modern_content, 2000)
        
        prompt = f"""Analyze the semantic differences between these two versions.

SECTION: {section_name}

LEGACY VERSION:
{legacy_content}

MODERNIZED VERSION:
{modern_content}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "section_name": "{section_name}",
  "key_differences": ["diff1", "diff2", "diff3"],
  "intent_shift": "description of intent change",
  "regulatory_impact": ["GDPR", "CCPA", "HIPAA", "None"],
  "risk_level": "Critical|High|Medium|Low",
  "business_impact": "brief impact description",
  "implementation_effort": "High|Medium|Low",
  "summary": "one line summary"
}}"""

        try:
            response = self._generate(prompt, max_tokens=400)
            result = json.loads(response)
            result['status'] = 'success'
            return result
        except json.JSONDecodeError:
            return {
                'section_name': section_name,
                'status': 'parse_error',
                'raw_response': response[:500]
            }
        except Exception as e:
            return {
                'section_name': section_name,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_removed_section(self, section_name: str, legacy_content: str) -> Dict:
        """Analyze removed section"""
        legacy_content = truncate_section(legacy_content, 2000)
        
        prompt = f"""This policy section was REMOVED.

SECTION: {section_name}

LEGACY CONTENT:
{legacy_content}

Respond ONLY with valid JSON:
{{
  "section_name": "{section_name}",
  "status": "REMOVED",
  "key_content": ["topic1", "topic2"],
  "regulatory_impact": ["GDPR", "CCPA", "HIPAA", "None"],
  "risk_level": "Critical|High|Medium|Low",
  "reason_for_removal": "likely reason",
  "compliance_implications": "what this means"
}}"""

        try:
            response = self._generate(prompt, max_tokens=300)
            result = json.loads(response)
            result['status'] = 'success'
            return result
        except Exception as e:
            return {
                'section_name': section_name,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_added_section(self, section_name: str, modern_content: str) -> Dict:
        """Analyze added section"""
        modern_content = truncate_section(modern_content, 2000)
        
        prompt = f"""This policy section is NEW.

SECTION: {section_name}

NEW CONTENT:
{modern_content}

Respond ONLY with valid JSON:
{{
  "section_name": "{section_name}",
  "status": "ADDED",
  "key_content": ["topic1", "topic2"],
  "regulatory_impact": ["GDPR", "CCPA", "HIPAA", "None"],
  "risk_level": "Critical|High|Medium|Low",
  "reason_for_addition": "likely reason",
  "implementation_requirement": "what needs to be done"
}}"""

        try:
            response = self._generate(prompt, max_tokens=300)
            result = json.loads(response)
            result['status'] = 'success'
            return result
        except Exception as e:
            return {
                'section_name': section_name,
                'status': 'error',
                'error': str(e)
            }
    
    def _generate(self, prompt: str, max_tokens: int = 400) -> str:
        """Generate response from Qwen"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    def analyze_batch(self, pairs: Dict) -> Dict:
        """Analyze all section pairs"""
        results = {
            'modified': [],
            'removed': [],
            'added': []
        }
        
        if pairs['modified']:
            print(f"\nAnalyzing {len(pairs['modified'])} modified sections...")
            for i, pair in enumerate(pairs['modified']):
                print(f"  [{i+1}/{len(pairs['modified'])}] {pair['section_name']}...", end='', flush=True)
                analysis = self.analyze_section_pair(
                    pair['section_name'],
                    pair['legacy_content'],
                    pair['modern_content']
                )
                results['modified'].append(analysis)
                print(" ✓")
        
        if pairs['removed']:
            print(f"\nAnalyzing {len(pairs['removed'])} removed sections...")
            for i, pair in enumerate(pairs['removed']):
                print(f"  [{i+1}/{len(pairs['removed'])}] {pair['section_name']}...", end='', flush=True)
                analysis = self.analyze_removed_section(
                    pair['section_name'],
                    pair['legacy_content']
                )
                results['removed'].append(analysis)
                print(" ✓")
        
        if pairs['added']:
            print(f"\nAnalyzing {len(pairs['added'])} added sections...")
            for i, pair in enumerate(pairs['added']):
                print(f"  [{i+1}/{len(pairs['added'])}] {pair['section_name']}...", end='', flush=True)
                analysis = self.analyze_added_section(
                    pair['section_name'],
                    pair['modern_content']
                )
                results['added'].append(analysis)
                print(" ✓")
        
        return results

print("Component 4: Qwen LLM Analyzer - READY")


# ============================================================================
# CELL 6: COMPONENT 5 - REPORT AGGREGATION
# ============================================================================

class ReportAggregator:
    def __init__(self):
        self.analysis_results = None
        self.comparison_metadata = None
    
    def aggregate(self, analysis_results: Dict, comparison_metadata: Dict) -> Dict:
        """Aggregate results into final report"""
        self.analysis_results = analysis_results
        self.comparison_metadata = comparison_metadata
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_modified_sections': len(analysis_results['modified']),
                'total_removed_sections': len(analysis_results['removed']),
                'total_added_sections': len(analysis_results['added']),
                'legacy_total_tokens': comparison_metadata.get('legacy_total_tokens'),
                'modern_total_tokens': comparison_metadata.get('modern_total_tokens'),
            },
            'risk_summary': self._calculate_risk_summary(analysis_results),
            'regulatory_impact': self._calculate_regulatory_impact(analysis_results),
            'modified_sections': analysis_results['modified'],
            'removed_sections': analysis_results['removed'],
            'added_sections': analysis_results['added'],
            'compliance_checklist': self._generate_checklist(analysis_results),
            'executive_summary': self._generate_summary(analysis_results)
        }
        
        return report
    
    def _calculate_risk_summary(self, analysis_results: Dict) -> Dict:
        """Calculate overall risk assessment"""
        risk_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for result in (analysis_results['modified'] + analysis_results['removed'] + analysis_results['added']):
            if result.get('status') == 'success':
                risk_level = result.get('risk_level', 'Medium')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        if risk_counts['Critical'] > 0:
            overall_risk = 'CRITICAL'
        elif risk_counts['High'] > 0:
            overall_risk = 'HIGH'
        elif risk_counts['Medium'] > 0:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        return {
            'overall_risk_level': overall_risk,
            'risk_breakdown': risk_counts
        }
    
    def _calculate_regulatory_impact(self, analysis_results: Dict) -> Dict:
        """Calculate regulatory framework impact"""
        regulatory_impact = {}
        
        all_results = (analysis_results['modified'] + 
                      analysis_results['removed'] + 
                      analysis_results['added'])
        
        for result in all_results:
            if result.get('status') == 'success':
                frameworks = result.get('regulatory_impact', [])
                if isinstance(frameworks, str):
                    frameworks = [frameworks]
                
                for framework in frameworks:
                    if framework not in ['None', '']:
                        if framework not in regulatory_impact:
                            regulatory_impact[framework] = {'count': 0, 'sections': []}
                        regulatory_impact[framework]['count'] += 1
                        regulatory_impact[framework]['sections'].append(result.get('section_name'))
        
        return regulatory_impact
    
    def _generate_checklist(self, analysis_results: Dict) -> List[Dict]:
        """Generate compliance checklist"""
        checklist = []
        
        all_results = (analysis_results['modified'] + 
                      analysis_results['removed'] + 
                      analysis_results['added'])
        
        for result in all_results:
            if result.get('status') == 'success':
                risk_level = result.get('risk_level', 'Medium')
                
                if risk_level in ['Critical', 'High']:
                    item = {
                        'section': result.get('section_name'),
                        'type': 'MODIFIED' if 'legacy_content' in result and 'modern_content' in result else result.get('status'),
                        'risk_level': risk_level,
                        'action': self._generate_action(result),
                        'owner': self._determine_owner(result)
                    }
                    checklist.append(item)
        
        return sorted(checklist, key=lambda x: ['Critical', 'High'].index(x['risk_level']))
    
    def _generate_action(self, result: Dict) -> str:
        """Generate recommended action"""
        if result.get('status') == 'REMOVED':
            return f"Review removal and assess impacts"
        elif result.get('status') == 'ADDED':
            return f"Implement: {result.get('implementation_requirement', 'Review')}"
        else:
            return f"Review and update implementation"
    
    def _determine_owner(self, result: Dict) -> str:
        """Determine team owner"""
        section_lower = result.get('section_name', '').lower()
        
        if any(x in section_lower for x in ['security', 'encrypt', 'protection']):
            return 'Security Team'
        elif any(x in section_lower for x in ['retention', 'deletion', 'data', 'privacy']):
            return 'Data Governance'
        elif any(x in section_lower for x in ['compliance', 'regulatory', 'gdpr']):
            return 'Legal/Compliance'
        else:
            return 'Engineering'
    
    def _generate_summary(self, analysis_results: Dict) -> str:
        """Generate executive summary"""
        modified_count = len(analysis_results['modified'])
        removed_count = len(analysis_results['removed'])
        added_count = len(analysis_results['added'])
        
        summary = f"Total Changes: {modified_count} modified, {removed_count} removed, {added_count} added\n\n"
        
        critical_sections = [r for r in (analysis_results['modified'] + 
                                         analysis_results['removed'] + 
                                         analysis_results['added'])
                            if r.get('risk_level') == 'Critical']
        
        if critical_sections:
            summary += "CRITICAL ITEMS:\n"
            for item in critical_sections[:3]:
                summary += f"  • {item.get('section_name')}\n"
        
        return summary
    
    def print_report(self, report: Dict):
        """Pretty print the report"""
        print(f"\n{'='*80}")
        print("POLICY COMPARISON REPORT - EXECUTIVE SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Generated: {report['metadata']['timestamp']}")
        print(f"Changes: {report['metadata']['total_modified_sections']} modified, " +
              f"{report['metadata']['total_removed_sections']} removed, " +
              f"{report['metadata']['total_added_sections']} added\n")
        
        risk = report['risk_summary']
        print(f"OVERALL RISK: {risk['overall_risk_level']}")
        print(f"  Critical: {risk['risk_breakdown']['Critical']}")
        print(f"  High:     {risk['risk_breakdown']['High']}")
        print(f"  Medium:   {risk['risk_breakdown']['Medium']}")
        print(f"  Low:      {risk['risk_breakdown']['Low']}\n")
        
        if report['regulatory_impact']:
            print("REGULATORY FRAMEWORKS IMPACTED:")
            for framework, data in report['regulatory_impact'].items():
                print(f"  • {framework}: {data['count']} changes")
        
        print(f"\nCOMPLIANCE CHECKLIST ({len(report['compliance_checklist'])} items):")
        for i, item in enumerate(report['compliance_checklist'][:5], 1):
            print(f"\n  [{i}] {item['section']} [{item['risk_level']}]")
            print(f"      Action: {item['action']}")
            print(f"      Owner: {item['owner']}")
        
        if len(report['compliance_checklist']) > 5:
            print(f"\n  ... and {len(report['compliance_checklist']) - 5} more items")
        
        print(f"\n{'='*80}\n")
    
    def export_json(self, report: Dict, filename: str = "policy_comparison_report.json"):
        """Export report as JSON"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report exported to {filename}")

print("Component 5: Report Aggregation - READY")


# ============================================================================
# CELL 7: MAIN PIPELINE
# ============================================================================

def run_pipeline(legacy_pdf: str, modern_pdf: str):
    """Execute complete pipeline"""
    print(f"\n{'='*80}")
    print("POLICY COMPARISON PIPELINE")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Step 1: Parse
    print("[1/6] Parsing PDFs with Docling...")
    legacy_markdown = parse_pdf(legacy_pdf)
    legacy_sections = extract_sections(legacy_markdown)
    legacy_tokens = print_structure(legacy_sections, "LEGACY POLICY")
    
    modern_markdown = parse_pdf(modern_pdf)
    modern_sections = extract_sections(modern_markdown)
    modern_tokens = print_structure(modern_sections, "MODERN POLICY")
    
    # Step 2: Build indexes
    print("[2/6] Building proxy indexes...")
    legacy_index = build_index(legacy_sections)
    modern_index = build_index(modern_sections)
    
    # Step 3: Diff detection
    print("[3/6] Detecting differences...")
    comparison = compare_indexes(legacy_index, modern_index)
    print_comparison(comparison)
    
    # Step 4: Retrieve content
    print("[4/6] Retrieving section content...")
    pairs = prepare_pairs(legacy_sections, modern_sections, comparison)
    print_pairs_summary(pairs)
    
    # Step 5: LLM Analysis
    print("[5/6] Initializing Qwen LLM...")
    analyzer = QwenAnalyzer()
    print("[5.5/6] Analyzing sections...")
    analysis_results = analyzer.analyze_batch(pairs)
    
    # Step 6: Aggregate
    print("[6/6] Generating report...")
    aggregator = ReportAggregator()
    report = aggregator.aggregate(analysis_results, comparison)
    aggregator.print_report(report)
    aggregator.export_json(report)
    
    elapsed = time.time() - start_time
    print(f"✓ Pipeline completed in {elapsed:.1f} seconds\n")
    
    return report

print("Main Pipeline - READY")


# ============================================================================
# CELL 8: EXECUTE PIPELINE
# ============================================================================

# Replace with your actual PDF paths
legacy_pdf = "legacy_policy.pdf"
modern_pdf = "modern_policy.pdf"

# Run the pipeline
report = run_pipeline(legacy_pdf, modern_pdf)


# ============================================================================
# CELL 9: VIEW DETAILED RESULTS
# ============================================================================

# View modified sections in detail
print("\n" + "="*80)
print("DETAILED MODIFIED SECTIONS")
print("="*80 + "\n")

for result in report['modified_sections'][:3]:  # First 3
    print(f"Section: {result.get('section_name')}")
    print(f"Risk: {result.get('risk_level')}")
    print(f"Regulatory: {result.get('regulatory_impact')}")
    print(f"Differences: {', '.join(result.get('key_differences', [])[:2])}")
    print(f"Summary: {result.get('summary')}")
    print("-" * 80 + "\n")