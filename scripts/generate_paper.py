#!/usr/bin/env python3
"""
Robust script to rebuild the main paper file from organized sections.

Uses token-based parsing instead of regex for reliable reference detection.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import mistune
import re
from dataclasses import dataclass
from enum import Enum


@dataclass
class HeaderInfo:
    """Complete information about a header."""
    section_num: str
    title: str
    anchor: str
    level: int
    line_number: Optional[int] = None


class TokenType(Enum):
    """Types of tokens we care about."""
    WORD = "word"
    SECTION_SYMBOL = "section_symbol"  # §
    NUMBER = "number"
    LETTER = "letter"
    DOT = "dot"
    DASH = "dash"
    COMMA = "comma"
    SEMICOLON = "semicolon"
    COLON = "colon"
    LPAREN = "lparen"
    RPAREN = "rparen"
    WHITESPACE = "whitespace"
    OTHER = "other"


@dataclass
class Token:
    """A token with its type and value."""
    type: TokenType
    value: str
    start: int
    end: int


class Tokenizer:
    """Tokenize text for reference detection."""
    
    def tokenize(self, text: str) -> List[Token]:
        """Break text into tokens."""
        tokens = []
        i = 0
        
        while i < len(text):
            # Check for section symbol
            if text[i] == '§':
                tokens.append(Token(TokenType.SECTION_SYMBOL, '§', i, i + 1))
                i += 1
            # Check for letters
            elif text[i].isalpha():
                start = i
                while i < len(text) and text[i].isalpha():
                    i += 1
                tokens.append(Token(TokenType.WORD if i - start > 1 else TokenType.LETTER, 
                                  text[start:i], start, i))
            # Check for digits
            elif text[i].isdigit():
                start = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                tokens.append(Token(TokenType.NUMBER, text[start:i], start, i))
            # Check for punctuation
            elif text[i] == '.':
                tokens.append(Token(TokenType.DOT, '.', i, i + 1))
                i += 1
            elif text[i] == '-' or text[i] == '–':
                tokens.append(Token(TokenType.DASH, text[i], i, i + 1))
                i += 1
            elif text[i] == ',':
                tokens.append(Token(TokenType.COMMA, ',', i, i + 1))
                i += 1
            elif text[i] == ';':
                tokens.append(Token(TokenType.SEMICOLON, ';', i, i + 1))
                i += 1
            elif text[i] == ':':
                tokens.append(Token(TokenType.COLON, ':', i, i + 1))
                i += 1
            elif text[i] == '(':
                tokens.append(Token(TokenType.LPAREN, '(', i, i + 1))
                i += 1
            elif text[i] == ')':
                tokens.append(Token(TokenType.RPAREN, ')', i, i + 1))
                i += 1
            elif text[i].isspace():
                start = i
                while i < len(text) and text[i].isspace():
                    i += 1
                tokens.append(Token(TokenType.WHITESPACE, text[start:i], start, i))
            else:
                tokens.append(Token(TokenType.OTHER, text[i], i, i + 1))
                i += 1
        
        return tokens


class ReferenceParser:
    """Parse section references from token streams."""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
    
    def parse_section_number(self, tokens: List[Token], start_idx: int) -> Optional[Tuple[str, int]]:
        """
        Try to parse a section number starting at start_idx.
        Returns (section_string, end_idx) or None.
        
        Valid formats:
        - "3" (single number)
        - "3.1" (number.number)
        - "A.1" (letter.number)
        - "A.1.2" (letter.number.number...)
        - "3.1.2" (number.number.number...)
        """
        if start_idx >= len(tokens):
            return None
        
        parts = []
        i = start_idx
        
        # First part: either LETTER or NUMBER
        if tokens[i].type == TokenType.NUMBER:
            parts.append(tokens[i].value)
            i += 1
        elif tokens[i].type == TokenType.LETTER and len(tokens[i].value) == 1:
            parts.append(tokens[i].value)
            i += 1
        else:
            return None
        
        # Subsequent parts: .NUMBER
        while i + 1 < len(tokens):
            if tokens[i].type == TokenType.DOT and tokens[i + 1].type == TokenType.NUMBER:
                parts.append('.')
                parts.append(tokens[i + 1].value)
                i += 2
            else:
                break
        
        if parts:
            return (''.join(parts), i)
        return None
    
    def find_references(self, text: str, headers: Dict[str, HeaderInfo],
                       current_section: Optional[str] = None) -> List[Tuple[int, int, str, str]]:
        """
        Find all references in text.
        Returns list of (start, end, original_text, linked_text).
        """
        tokens = self.tokenizer.tokenize(text)
        references = []
        i = 0

        while i < len(tokens):
            result = self._try_parse_reference(tokens, i, headers, current_section)
            if result:
                start, end, end_token_idx, original, linked = result
                references.append((start, end, original, linked))
                i = end_token_idx
            else:
                i += 1

        return references
    
    def _try_parse_reference(self, tokens: List[Token], start_idx: int,
                            headers: Dict[str, HeaderInfo],
                            current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """
        Try to parse a reference starting at start_idx.
        Returns (start_pos, end_pos, end_token_idx, original_text, linked_text) or None.
        """
        if start_idx >= len(tokens):
            return None
        
        token = tokens[start_idx]
        
        # Case 1: § or §§ followed by section number
        if token.type == TokenType.SECTION_SYMBOL:
            return self._parse_section_symbol_ref(tokens, start_idx, headers, current_section)
        
        # Case 2: "Section" or "section" followed by number
        if token.type == TokenType.WORD and token.value.lower() == 'section':
            return self._parse_section_word_ref(tokens, start_idx, headers, current_section)
        
        # Case 3: "Appendix" or "appendix" followed by ref
        if token.type == TokenType.WORD and token.value.lower() == 'appendix':
            return self._parse_appendix_word_ref(tokens, start_idx, headers, current_section)
        
        # Case 4: "App" or "App." followed by ref
        if token.type == TokenType.WORD and token.value.lower() == 'app':
            return self._parse_app_ref(tokens, start_idx, headers, current_section)
        
        # Case 5: Lemma reference
        if token.type == TokenType.WORD and token.value.lower() == 'lemma':
            return self._parse_lemma_ref(tokens, start_idx, headers, current_section)

        # Case 6: Theorem reference
        if token.type == TokenType.WORD and token.value.lower() == 'theorem':
            return self._parse_theorem_ref(tokens, start_idx, headers, current_section)

        # Case 7: Proposition reference
        if token.type == TokenType.WORD and token.value.lower() == 'proposition':
            return self._parse_proposition_ref(tokens, start_idx, headers, current_section)

        # Case 8: Definition reference
        if token.type == TokenType.WORD and token.value.lower() == 'definition':
            return self._parse_definition_ref(tokens, start_idx, headers, current_section)

        # Case 9: Standalone appendix reference (Letter.Number)
        # Only if preceded by certain contexts
        if token.type == TokenType.LETTER and len(token.value) == 1:
            return self._parse_standalone_appendix_ref(tokens, start_idx, headers, current_section)

        return None
    
    def _parse_section_symbol_ref(self, tokens: List[Token], start_idx: int,
                                  headers: Dict[str, HeaderInfo],
                                  current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse §X.Y or §§X.Y references."""
        i = start_idx
        prefix = '§'
        
        # Check for double §
        if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.SECTION_SYMBOL:
            prefix = '§§'
            i += 1
        
        i += 1  # Move past the §
        
        # Skip optional whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1
        
        # Try to parse section number
        result = self.parse_section_number(tokens, i)
        if not result:
            return None
        
        section_ref, end_idx = result
        
        # Check for range (- or –)
        if end_idx < len(tokens) and tokens[end_idx].type == TokenType.DASH:
            # Try to parse second section number
            result2 = self.parse_section_number(tokens, end_idx + 1)
            if result2:
                end_section, end_idx = result2
                # Build the range reference
                original_text = self._get_text_between(tokens, start_idx, end_idx)

                # Link both parts if valid
                start_link = self._make_link(section_ref, headers, current_section)
                end_link = self._make_link(end_section, headers, current_section)

                if start_link and end_link:
                    linked_text = f'[{prefix}{section_ref}]({start_link})-[{end_section}]({end_link})'
                else:
                    linked_text = original_text

                return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        # Single reference
        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            linked_text = f'[{prefix}{section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        return None
    
    def _parse_section_word_ref(self, tokens: List[Token], start_idx: int,
                               headers: Dict[str, HeaderInfo],
                               current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Section X.Y' references."""
        i = start_idx + 1  # Skip 'Section'
        
        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1
        
        result = self.parse_section_number(tokens, i)
        if not result:
            return None
        
        section_ref, end_idx = result
        
        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            linked_text = f'[Section {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        return None
    
    def _parse_appendix_word_ref(self, tokens: List[Token], start_idx: int,
                                headers: Dict[str, HeaderInfo],
                                current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Appendix X.Y' references."""
        i = start_idx + 1  # Skip 'Appendix'
        
        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1
        
        result = self.parse_section_number(tokens, i)
        if not result:
            return None
        
        section_ref, end_idx = result
        
        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            linked_text = f'[Appendix {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        return None
    
    def _parse_app_ref(self, tokens: List[Token], start_idx: int,
                      headers: Dict[str, HeaderInfo],
                      current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'App X.Y' or 'App. X.Y' references."""
        i = start_idx + 1  # Skip 'App'
        has_period = False
        
        # Check for period
        if i < len(tokens) and tokens[i].type == TokenType.DOT:
            has_period = True
            i += 1
        
        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1
        
        result = self.parse_section_number(tokens, i)
        if not result:
            return None

        section_ref, end_idx = result

        # Only match appendix-style references (starting with letter)
        if not section_ref[0].isupper():
            return None

        # Check for range (- or –)
        if end_idx < len(tokens) and tokens[end_idx].type == TokenType.DASH:
            # Try to parse second section number
            result2 = self.parse_section_number(tokens, end_idx + 1)
            if result2:
                end_section, end_idx = result2
                # Check if end_section is also appendix-style
                if end_section[0].isupper():
                    # Build the range reference
                    original_text = self._get_text_between(tokens, start_idx, end_idx)

                    # Link both parts if valid
                    start_link = self._make_link(section_ref, headers, current_section)
                    end_link = self._make_link(end_section, headers, current_section)

                    prefix = 'App.' if has_period else 'App'
                    if start_link and end_link:
                        linked_text = f'[{prefix} {section_ref}]({start_link})-[{end_section}]({end_link})'
                    else:
                        linked_text = original_text

                    return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        # Single reference
        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            prefix = 'App.' if has_period else 'App'
            linked_text = f'[{prefix} {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        return None

    def _parse_lemma_ref(self, tokens: List[Token], start_idx: int,
                        headers: Dict[str, HeaderInfo],
                        current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Lemma A.X.Y' references."""
        lemma_word = tokens[start_idx].value
        i = start_idx + 1  # Skip 'Lemma'

        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1

        result = self.parse_section_number(tokens, i)
        if not result:
            return None

        section_ref, end_idx = result

        # Only match appendix-style references (starting with uppercase letter)
        if not section_ref[0].isupper():
            return None

        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            # Preserve the original capitalization
            linked_text = f'[{lemma_word} {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        return None

    def _parse_theorem_ref(self, tokens: List[Token], start_idx: int,
                          headers: Dict[str, HeaderInfo],
                          current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Theorem A.X.Y' references."""
        theorem_word = tokens[start_idx].value
        i = start_idx + 1  # Skip 'Theorem'

        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1

        result = self.parse_section_number(tokens, i)
        if not result:
            return None

        section_ref, end_idx = result

        # Only match appendix-style references (starting with uppercase letter)
        if not section_ref[0].isupper():
            return None

        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            # Preserve the original capitalization
            linked_text = f'[{theorem_word} {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        return None
        """Parse 'Theorem X.Y' or 'Lemma X.Y' references."""
        theorem_word = tokens[start_idx].value
        i = start_idx + 1  # Skip the theorem word
        
        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1
        
        result = self.parse_section_number(tokens, i)
        if not result:
            return None
        
        section_ref, end_idx = result
        
        # Only match appendix-style references (starting with letter)
        if not section_ref[0].isupper():
            return None
        
        if self._is_valid_reference(section_ref, headers, current_section):
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            anchor = headers[section_ref].anchor
            # Preserve the original capitalization
            linked_text = f'[{theorem_word} {section_ref}](#{anchor})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        return None
    
    def _parse_standalone_appendix_ref(self, tokens: List[Token], start_idx: int,
                                      headers: Dict[str, HeaderInfo],
                                      current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """
        Parse standalone appendix references like 'A.1.2'.
        Only match if preceded by specific contexts (parentheses, colons, commas).
        Also handles cases like "(A.5.1)" or after "cf." or "via"
        """
        # Check what comes before - need to look back through whitespace
        preceding_idx = start_idx - 1
        
        # Skip backwards through whitespace to find the actual preceding token
        while preceding_idx >= 0 and tokens[preceding_idx].type == TokenType.WHITESPACE:
            preceding_idx -= 1
        
        if preceding_idx >= 0:
            prev_token = tokens[preceding_idx]
            
            # Allow after these punctuation marks
            if prev_token.type in (TokenType.LPAREN, TokenType.COLON, 
                                  TokenType.COMMA, TokenType.SEMICOLON):
                pass  # Good context
            # Also allow after words like "cf.", "via", "see", and other common reference contexts
            elif prev_token.type == TokenType.WORD and prev_token.value.lower() in ['cf', 'via', 'see', 'as', 'in', 'from', 'by', 'using', 'for', 'with', 'to', 'of', 'on', 'at', 'closure', 'structure', 'bound', 'test', 'result', 'proof', 'theorem', 'proposition', 'lemma', 'definition']:
                pass  # Good context
            # Also allow after "App." or "Appendix" (for cases already handled by other parsers)
            elif prev_token.type == TokenType.WORD and prev_token.value.lower() in ['app', 'appendix']:
                return None  # Let the specific parsers handle these
            else:
                return None  # Not a valid context
        
        result = self.parse_section_number(tokens, start_idx)
        if not result:
            return None
        
        section_ref, end_idx = result
        
        # Only match if it looks like an appendix reference (starts with uppercase letter)
        # and has at least one dot (e.g., "A.1", not just "A")
        if not section_ref[0].isupper() or '.' not in section_ref:
            return None
        
        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            linked_text = f'[{section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)
        
        return None

    def _parse_proposition_ref(self, tokens: List[Token], start_idx: int,
                              headers: Dict[str, HeaderInfo],
                              current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Proposition A.X.Y' references."""
        proposition_word = tokens[start_idx].value
        i = start_idx + 1  # Skip 'Proposition'

        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1

        result = self.parse_section_number(tokens, i)
        if not result:
            return None

        section_ref, end_idx = result

        # Only match appendix-style references (starting with uppercase letter)
        if not section_ref[0].isupper():
            return None

        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            # Preserve the original capitalization
            linked_text = f'[{proposition_word} {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        return None

    def _parse_definition_ref(self, tokens: List[Token], start_idx: int,
                             headers: Dict[str, HeaderInfo],
                             current_section: Optional[str]) -> Optional[Tuple[int, int, int, str, str]]:
        """Parse 'Definition A.X.Y' references."""
        definition_word = tokens[start_idx].value
        i = start_idx + 1  # Skip 'Definition'

        # Skip whitespace
        while i < len(tokens) and tokens[i].type == TokenType.WHITESPACE:
            i += 1

        result = self.parse_section_number(tokens, i)
        if not result:
            return None

        section_ref, end_idx = result

        # Only match appendix-style references (starting with uppercase letter)
        if not section_ref[0].isupper():
            return None

        link = self._make_link(section_ref, headers, current_section)
        if link:
            original_text = self._get_text_between(tokens, start_idx, end_idx)
            # Preserve the original capitalization
            linked_text = f'[{definition_word} {section_ref}]({link})'
            return (tokens[start_idx].start, tokens[end_idx - 1].end, end_idx, original_text, linked_text)

        return None

    def _is_valid_reference(self, ref: str, headers: Dict[str, HeaderInfo],
                           current_section: Optional[str]) -> bool:
        """Check if reference is valid and not self-reference."""
        # First check if the exact reference exists
        if ref in headers:
            return self._check_self_reference(ref, current_section)

        # If not, try to find a parent section (for references like A.2.2.3 -> A.2.2)
        parent_ref = self._find_parent_reference(ref, headers)
        if parent_ref:
            return self._check_self_reference(parent_ref, current_section)

        return False

    def _find_parent_reference(self, ref: str, headers: Dict[str, HeaderInfo]) -> Optional[str]:
        """Find the parent section for a reference like A.2.2.3 -> A.2.2"""
        parts = ref.split('.')
        if len(parts) <= 1:
            return None

        # Try removing parts from the end until we find a matching header
        for i in range(len(parts) - 1, 0, -1):
            parent_ref = '.'.join(parts[:i])
            if parent_ref in headers:
                return parent_ref

        return None

    def _check_self_reference(self, ref: str, current_section: Optional[str]) -> bool:
        """Check if reference is a self-reference and should not be linked."""
        if not current_section:
            return True

        # Extract section number from current section directory name
        match = re.match(r'^(\d+)', current_section)
        current_num = match.group(1) if match else None

        # Don't link if it's a self-reference
        if current_num and (ref == current_num or ref.startswith(current_num + '.')):
            return False

        return True
    
    def _make_link(self, ref: str, headers: Dict[str, HeaderInfo],
                  current_section: Optional[str]) -> Optional[str]:
        """Make a link anchor for a reference, or None if invalid."""
        if self._is_valid_reference(ref, headers, current_section):
            # Use the exact reference if it exists
            if ref in headers:
                return f"#{headers[ref].anchor}"
            # Otherwise use the parent reference
            parent_ref = self._find_parent_reference(ref, headers)
            if parent_ref:
                return f"#{headers[parent_ref].anchor}"
        return None
    
    def _get_text_between(self, tokens: List[Token], start_idx: int, end_idx: int) -> str:
        """Get the original text between two token indices."""
        if start_idx >= end_idx or start_idx >= len(tokens):
            return ""
        
        start_pos = tokens[start_idx].start
        end_pos = tokens[end_idx - 1].end
        
        # Reconstruct from tokens (this preserves exact spacing)
        parts = []
        for i in range(start_idx, end_idx):
            parts.append(tokens[i].value)
        return ''.join(parts)


class PaperConfig:
    """Configuration for paper building process."""
    
    def __init__(self):
        self.sections_dir = Path("docs/paper/sections")
        self.output_file = Path("docs/paper/A Mathematical Theory of Contradiction.md")


class RobustMarkdownParser:
    """Uses mistune AST for reliable markdown parsing."""
    
    def __init__(self):
        self.markdown = mistune.create_markdown(renderer='ast')
    
    def parse_document(self, content: str) -> List[dict]:
        """Parse markdown into AST."""
        return self.markdown(content)
    
    def extract_headers_from_ast(self, ast_nodes: List[dict], parent_nums: List[int] = None) -> List[HeaderInfo]:
        """Extract headers by walking the AST."""
        if parent_nums is None:
            parent_nums = []
        
        headers = []
        
        # Handle both list and dict inputs
        if isinstance(ast_nodes, dict):
            ast_nodes = [ast_nodes]
        
        for node in ast_nodes:
            if not isinstance(node, dict):
                continue
                
            node_type = node.get('type', '')
            
            if node_type == 'heading':
                level = node.get('attrs', {}).get('level', node.get('level', 1))
                # Extract plain text from heading children
                title = self._extract_text_from_nodes(node.get('children', []))
                
                # Try to extract section number from title
                section_num = self._extract_section_from_title(title)
                
                if section_num:
                    anchor = self.create_anchor(title)
                    headers.append(HeaderInfo(
                        section_num=section_num,
                        title=title,
                        anchor=anchor,
                        level=level
                    ))
            
            # Recursively process children
            if 'children' in node:
                headers.extend(self.extract_headers_from_ast(node['children'], parent_nums))
        
        return headers
    
    def _extract_text_from_nodes(self, nodes: List[dict]) -> str:
        """Recursively extract plain text from AST nodes."""
        if not nodes:
            return ""
        
        text_parts = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
                
            node_type = node.get('type', '')
            
            if node_type == 'text':
                text_parts.append(node.get('raw', ''))
            elif node_type in ('strong', 'emphasis', 'codespan', 'link'):
                text_parts.append(self._extract_text_from_nodes(node.get('children', [])))
            elif 'children' in node:
                text_parts.append(self._extract_text_from_nodes(node['children']))
        
        return ''.join(text_parts)
    
    def _extract_section_from_title(self, title: str) -> Optional[str]:
        """
        Extract section number from title text.
        Handles formats like:
        - "1" or "3" (single digit sections)
        - "1.2 Title"
        - "A.3.1 Title"
        - "A.1.2.3 Title" (multi-level appendix)
        - "Definition A.3.1 (Agreement and Contradiction)."
        - "Appendix B"
        """
        title = title.strip()

        # Special case: handle "Appendix X" patterns
        appendix_match = re.search(r'Appendix\s+([A-Z])', title, re.IGNORECASE)
        if appendix_match:
            return appendix_match.group(1)

        # Try to find section numbers anywhere in the title
        # Look for patterns like A.3.1, 1.2.3, etc.
        # Find all matches and take the longest one
        matches = re.findall(r'([A-Z](?:\.\d+)+|\d+(?:\.\d+)*)', title)
        if matches:
            # Take the longest match (most specific)
            return max(matches, key=len)

        # Try single letter (for appendices without subsections)
        match = re.match(r'^([A-Z])(?:\s|$|\.)', title)
        if match:
            return match.group(1)

        return None
    
    def create_anchor(self, text: str) -> str:
        """Create GitHub-style anchor from header text."""
        # Strip markdown and convert to lowercase
        text = text.lower()
        # Remove special characters, keep alphanumeric and spaces/hyphens
        text = re.sub(r'[^\w\s-]', '', text)
        # Replace spaces with hyphens
        text = re.sub(r'\s+', '-', text)
        # Remove leading/trailing hyphens and collapse multiple hyphens
        text = re.sub(r'-+', '-', text).strip('-')
        return text


class SectionTracker:
    """Explicitly tracks which section each content line belongs to."""
    
    def __init__(self):
        self.section_map: Dict[int, str] = {}  # line_number -> section_name
        self.current_section: Optional[str] = None
    
    def set_section(self, section_name: str, start_line: int, end_line: int):
        """Mark a range of lines as belonging to a section."""
        for line_num in range(start_line, end_line + 1):
            self.section_map[line_num] = section_name
    
    def get_section_for_line(self, line_num: int) -> Optional[str]:
        """Get the section name for a given line number."""
        return self.section_map.get(line_num)


class RobustCrossReferenceHandler:
    """Token-based cross-reference handler."""
    
    def __init__(self):
        self.parser = ReferenceParser()
    
    def add_cross_references(
        self, 
        content: str, 
        headers: Dict[str, HeaderInfo],
        section_tracker: SectionTracker
    ) -> str:
        """Add cross-references using token-based parsing."""
        lines = content.split('\n')
        processed_lines = []
        
        for line_num, line in enumerate(lines):
            # Skip header lines
            if line.strip().startswith('#'):
                processed_lines.append(line)
                continue
            
            # Skip lines that are already links (contain [...](...))
            if re.search(r'\[([^\]]+)\]\([^)]+\)', line):
                # Still process, but be careful
                pass
            
            # Get current section
            current_section = section_tracker.get_section_for_line(line_num)
            
            # Find all references in the line
            references = self.parser.find_references(line, headers, current_section)
            
            # Apply replacements in reverse order to preserve positions
            references.sort(key=lambda x: x[0], reverse=True)
            
            result = line
            for start, end, original, linked in references:
                # Only replace if we're not inside an existing link
                if not self._is_inside_link(result, start):
                    result = result[:start] + linked + result[end:]
            
            processed_lines.append(result)
        
        processed_content = '\n'.join(processed_lines)

        # Validate generated links
        self._validate_generated_links(processed_content, headers)

        return processed_content
    
    def _is_inside_link(self, text: str, position: int) -> bool:
        """Check if a position is inside an existing markdown link."""
        # Find all existing links
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        for match in link_pattern.finditer(text):
            if match.start() <= position < match.end():
                return True
        return False

    def _validate_generated_links(self, content: str, headers: Dict[str, HeaderInfo]) -> None:
        """Validate that all generated markdown links point to existing anchors."""
        import re

        # Find all markdown links: [text](url)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        anchor_pattern = re.compile(r'^#(.+)$')

        existing_anchors = {header.anchor for header in headers.values()}

        warnings = []
        for match in link_pattern.finditer(content):
            url = match.group(2)
            anchor_match = anchor_pattern.match(url)
            if anchor_match:
                anchor = anchor_match.group(1)
                if anchor not in existing_anchors:
                    link_text = match.group(1)
                    warnings.append(f"  WARNING: Link '{link_text}' points to non-existent anchor '#{anchor}'")

        if warnings:
            print(f"\n⚠️  Found {len(warnings)} generated links that point to non-existent anchors:")
            for warning in warnings:
                print(warning)
        else:
            print("\n✅ All generated links point to valid anchors")


class FileHandler:
    """Handles file operations."""
    
    def __init__(self, config: PaperConfig):
        self.config = config
    
    def get_section_directories(self) -> List[Path]:
        """Get all section directories in alphabetical order."""
        section_dirs = [d for d in self.config.sections_dir.iterdir() if d.is_dir()]
        return sorted(section_dirs, key=lambda x: x.name)
    
    def read_section_content(self, section_dir: Path) -> str:
        """Read all markdown files in a section directory."""
        md_files = sorted(section_dir.glob("*.md"))
        
        if not md_files:
            raise FileNotFoundError(f"No .md files found in {section_dir.name}")
        
        content_parts = []
        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            content_parts.append(content)
        
        return "\n\n".join(content_parts)


class PaperBuilder:
    """Main builder class."""
    
    def __init__(self, config: PaperConfig):
        self.config = config
        self.parser = RobustMarkdownParser()
        self.file_handler = FileHandler(config)
        self.cross_ref_handler = RobustCrossReferenceHandler()
        self.section_tracker = SectionTracker()
    
    def build(self):
        """Main build process."""
        print("Starting paper rebuild with token-based parsing...")
        
        # Get section directories
        section_dirs = self.file_handler.get_section_directories()
        print(f"Found {len(section_dirs)} sections")
        
        # Pass 1: Collect all content and track sections
        content_parts = []
        current_line = 0
        
        for section_dir in section_dirs:
            print(f"  Reading: {section_dir.name}")
            content = self.file_handler.read_section_content(section_dir)
            
            # Track which lines belong to this section
            num_lines = content.count('\n') + 1
            self.section_tracker.set_section(
                section_dir.name,
                current_line,
                current_line + num_lines - 1
            )
            
            content_parts.append(content)
            current_line += num_lines + 2  # +2 for the newlines we'll add between sections
        
        # Combine all content
        full_content = "\n\n".join(content_parts)
        
        # Pass 2: Extract all headers
        print("\nExtracting headers...")
        ast = self.parser.parse_document(full_content)
        headers_list = self.parser.extract_headers_from_ast(ast)
        
        # Create lookup dictionary, preferring section headers over theorem/lemma headers
        headers_dict = {}
        for h in headers_list:
            if h.section_num:
                if h.section_num not in headers_dict:
                    headers_dict[h.section_num] = h
                else:
                    # If there's a conflict, prefer the header that looks more like a section
                    existing = headers_dict[h.section_num]
                    # Prefer headers that don't contain theorem/lemma keywords
                    if re.search(r'\b(Theorem|Lemma|Proposition|Definition|Corollary)\b', existing.title, re.IGNORECASE):
                        if not re.search(r'\b(Theorem|Lemma|Proposition|Definition|Corollary)\b', h.title, re.IGNORECASE):
                            headers_dict[h.section_num] = h  # Prefer the non-theorem header
                    # Or prefer lower level headers
                    elif h.level < existing.level:
                        headers_dict[h.section_num] = h

        print(f"  Found {len(headers_dict)} section headers")
        
        # Pass 3: Add cross-references using token-based parsing
        print("\nAdding cross-references with token-based parser...")
        processed_content = self.cross_ref_handler.add_cross_references(
            full_content,
            headers_dict,
            self.section_tracker
        )
        
        # Write output
        print("\nWriting output...")
        self._write_output(processed_content, section_dirs)
        
        print(f"\n✓ Successfully built paper at {self.config.output_file}")
    
    def _write_output(self, content: str, sections: List[Path]):
        """Write final content with header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        header = f"""<!--
This file was automatically generated from organized sections in docs/paper/sections/
using the rebuild_paper.py script.

Generated on: {timestamp}

To regenerate this file after editing sections:
    python scripts/rebuild_paper.py

Sections are organized alphabetically in the following order:
"""
        
        for i, section_dir in enumerate(sections, 1):
            header += f"    {i}. {section_dir.name}\n"
        
        header += "-->\n\n"
        
        final_content = header + content
        
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_file.write_text(final_content, encoding='utf-8')


def main():
    """Entry point."""
    try:
        config = PaperConfig()
        builder = PaperBuilder(config)
        builder.build()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()