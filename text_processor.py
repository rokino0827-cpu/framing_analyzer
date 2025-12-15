"""
文本预处理模块 - Step 1 & 2
包含噪声清理、句子切分、结构区划分
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedArticle:
    """预处理后的文章结构"""
    title: str
    content: str
    sentences: List[str]
    zones: Dict[str, List[str]]  # {"headline": [...], "lede": [...], "quotes": [...], "narration": [...]}
    sentence_positions: List[Tuple[int, int]]  # 每个句子在原文中的位置 (start, end)

class TextProcessor:
    """文本预处理器 - Step 1"""
    
    def __init__(self, config):
        self.config = config.processing
        
        # 编译正则表达式
        self.noise_pattern = self._compile_noise_pattern()
        self.sentence_patterns = self._compile_sentence_patterns()
    
    def _compile_noise_pattern(self) -> re.Pattern:
        """编译噪声过滤模式"""
        keywords = '|'.join(re.escape(kw) for kw in self.config.noise_keywords)
        return re.compile(f'({keywords})', re.IGNORECASE)
    
    def _compile_sentence_patterns(self) -> Dict[str, re.Pattern]:
        """编译句子切分模式"""
        patterns = {}
        
        # 中文句子结束符
        chinese_endings = '|'.join(re.escape(end) for end in self.config.chinese_sentence_endings)
        patterns['chinese'] = re.compile(f'([^{chinese_endings}]*[{chinese_endings}])')
        
        # 英文句子结束符（简化版，不处理缩写）
        english_endings = '|'.join(re.escape(end) for end in self.config.english_sentence_endings)
        patterns['english'] = re.compile(f'([^{english_endings}]*[{english_endings}])')
        
        return patterns
    
    def clean_noise_paragraphs(self, text: str) -> str:
        """清理噪声段落"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否包含噪声关键词
            if self.noise_pattern.search(line):
                logger.debug(f"Filtered noise line: {line[:50]}...")
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """规范空白与换行"""
        # 统一换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # 移除多余空白
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 规范多个换行为最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_sentences(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """句子切分（中英文混合）"""
        sentences = []
        positions = []
        
        # 按换行分段
        paragraphs = text.split('\n')
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                current_pos += 1  # 换行符
                continue
            
            # 检测语言并切分句子
            para_sentences, para_positions = self._split_paragraph_sentences(
                paragraph, current_pos
            )
            
            sentences.extend(para_sentences)
            positions.extend(para_positions)
            
            current_pos += len(paragraph) + 1  # 段落长度 + 换行符
        
        return sentences, positions
    
    def _split_paragraph_sentences(self, paragraph: str, start_pos: int) -> Tuple[List[str], List[Tuple[int, int]]]:
        """切分段落中的句子"""
        sentences = []
        positions = []
        
        # 简化策略：按标点符号切分
        # 中文标点
        chinese_pattern = r'([^。！？…；]*[。！？…；])'
        # 英文标点
        english_pattern = r'([^.!?]*[.!?]+)'
        
        # 先尝试中文切分
        chinese_matches = re.findall(chinese_pattern, paragraph)
        if chinese_matches and len(''.join(chinese_matches).strip()) > len(paragraph) * 0.5:
            # 主要是中文
            current_pos = start_pos
            for match in chinese_matches:
                match = match.strip()
                if match:
                    sentences.append(match)
                    positions.append((current_pos, current_pos + len(match)))
                    current_pos += len(match)
        else:
            # 主要是英文或混合，按英文规则切分
            english_matches = re.findall(english_pattern, paragraph)
            if english_matches:
                current_pos = start_pos
                for match in english_matches:
                    match = match.strip()
                    if match:
                        sentences.append(match)
                        positions.append((current_pos, current_pos + len(match)))
                        current_pos += len(match)
            else:
                # 没有明显的句子结束符，整段作为一句
                if paragraph.strip():
                    sentences.append(paragraph.strip())
                    positions.append((start_pos, start_pos + len(paragraph)))
        
        return sentences, positions
    
    def process_article(self, content: str, title: str = "") -> ProcessedArticle:
        """处理单篇文章"""
        logger.debug(f"Processing article: {title[:50]}...")
        
        # Step 1: 文本预处理
        cleaned_content = self.clean_noise_paragraphs(content)
        normalized_content = self.normalize_whitespace(cleaned_content)
        
        # 句子切分
        sentences, positions = self.split_sentences(normalized_content)
        
        logger.debug(f"Split into {len(sentences)} sentences")
        
        return ProcessedArticle(
            title=title.strip(),
            content=normalized_content,
            sentences=sentences,
            zones={},  # 将由StructureZoneExtractor填充
            sentence_positions=positions
        )

class StructureZoneExtractor:
    """结构区划分器 - Step 2"""
    
    def __init__(self, config):
        self.config = config.processing
        
        # 编译引号模式
        self.quote_patterns = [re.compile(pattern) for pattern in self.config.quote_patterns]
    
    def extract_quotes(self, sentences: List[str]) -> List[int]:
        """提取引号句子的索引"""
        quote_indices = set()
        
        for i, sentence in enumerate(sentences):
            for pattern in self.quote_patterns:
                if pattern.search(sentence):
                    quote_indices.add(i)
                    break
        
        return list(quote_indices)
    
    def divide_into_zones(self, article: ProcessedArticle) -> ProcessedArticle:
        """将文章划分为结构区"""
        sentences = article.sentences
        
        # Z1: Headline (标题)
        headline_sentences = [article.title] if article.title else []
        
        # Z2: Lede (导语 - 正文前N句)
        lede_count = min(self.config.lede_sentence_count, len(sentences))
        lede_sentences = sentences[:lede_count]
        
        # Z3: Quotes (引号句子)
        quote_indices = self.extract_quotes(sentences)
        quote_sentences = [sentences[i] for i in quote_indices]
        
        # Z4: Narration (正文句子 - 去掉引号句子)
        narration_sentences = [
            sentences[i] for i in range(len(sentences)) 
            if i not in quote_indices
        ]
        
        # 更新zones
        article.zones = {
            "headline": headline_sentences,
            "lede": lede_sentences,
            "quotes": quote_sentences,
            "narration": narration_sentences
        }
        
        logger.debug(f"Zone division - Headline: {len(headline_sentences)}, "
                    f"Lede: {len(lede_sentences)}, "
                    f"Quotes: {len(quote_sentences)}, "
                    f"Narration: {len(narration_sentences)}")
        
        return article

class FragmentGenerator:
    """片段生成器 - Step 3"""
    
    def __init__(self, config):
        self.config = config.teacher
        
    def create_fragments(self, article: ProcessedArticle) -> List[Dict]:
        """将结构区转换为512 token可处理的片段"""
        fragments = []
        
        for zone_name, sentences in article.zones.items():
            if not sentences:
                continue
            
            if self.config.fragment_mode == "sentence":
                # 模式A: 句子级评分
                zone_fragments = self._create_sentence_fragments(sentences, zone_name)
            else:
                # 模式B: 块级评分
                zone_fragments = self._create_chunk_fragments(sentences, zone_name)
            
            fragments.extend(zone_fragments)
        
        return fragments
    
    def _create_sentence_fragments(self, sentences: List[str], zone: str) -> List[Dict]:
        """创建句子级片段"""
        fragments = []
        
        for i, sentence in enumerate(sentences):
            # 简单的token估算（1个字符约等于0.5个token）
            estimated_tokens = len(sentence) // 2
            
            if estimated_tokens <= self.config.max_length:
                # 句子长度合适，直接使用
                fragments.append({
                    'text': sentence,
                    'zone': zone,
                    'sentence_idx': i,
                    'fragment_type': 'sentence',
                    'estimated_tokens': estimated_tokens
                })
            else:
                # 句子过长，使用滑窗切分
                sliding_fragments = self._create_sliding_window_fragments(
                    sentence, zone, i
                )
                fragments.extend(sliding_fragments)
        
        return fragments
    
    def _create_chunk_fragments(self, sentences: List[str], zone: str) -> List[Dict]:
        """创建块级片段"""
        fragments = []
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        for i, sentence in enumerate(sentences):
            estimated_tokens = len(sentence) // 2
            
            if current_tokens + estimated_tokens <= self.config.max_length:
                # 可以加入当前块
                current_chunk.append(sentence)
                current_tokens += estimated_tokens
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    fragments.append({
                        'text': ' '.join(current_chunk),
                        'zone': zone,
                        'chunk_idx': chunk_idx,
                        'fragment_type': 'chunk',
                        'sentence_range': (len(fragments), i),
                        'estimated_tokens': current_tokens
                    })
                    chunk_idx += 1
                
                # 开始新块
                current_chunk = [sentence]
                current_tokens = estimated_tokens
        
        # 保存最后一个块
        if current_chunk:
            fragments.append({
                'text': ' '.join(current_chunk),
                'zone': zone,
                'chunk_idx': chunk_idx,
                'fragment_type': 'chunk',
                'sentence_range': (len(fragments), len(sentences)),
                'estimated_tokens': current_tokens
            })
        
        return fragments
    
    def _create_sliding_window_fragments(self, sentence: str, zone: str, sentence_idx: int) -> List[Dict]:
        """为超长句子创建滑窗片段"""
        fragments = []
        
        # 简单的字符级滑窗（实际应该用tokenizer，但这里简化）
        window_size = self.config.sliding_window_size * 2  # 字符数近似
        stride = self.config.sliding_window_stride * 2
        
        start = 0
        window_idx = 0
        
        while start < len(sentence):
            end = min(start + window_size, len(sentence))
            window_text = sentence[start:end]
            
            fragments.append({
                'text': window_text,
                'zone': zone,
                'sentence_idx': sentence_idx,
                'window_idx': window_idx,
                'fragment_type': 'sliding_window',
                'char_range': (start, end),
                'estimated_tokens': len(window_text) // 2
            })
            
            if end >= len(sentence):
                break
            
            start += stride
            window_idx += 1
        
        return fragments