#!/usr/bin/env python
"""Clean Project Gutenberg texts for fine-tuning.

Removes headers, footers, boilerplate, and prepares text for LLM training.

Usage:
    python clean_dataset.py --input-dir dataset --output-dir cleaned_dataset
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg boilerplate and clean the text.
    
    Args:
        text: Raw text from a Gutenberg file
        
    Returns:
        Cleaned text containing only the story content
    """
    # Find the start marker
    start_patterns = [
        r'\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]
    
    start_pos = -1
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break
    
    # Find the end marker
    end_patterns = [
        r'\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]
    
    end_pos = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break
    
    # Extract content between markers
    if start_pos == -1:
        # No start marker found, try to skip initial boilerplate
        lines = text.split('\n')
        # Look for where the actual content might start
        for i, line in enumerate(lines):
            if i > 50:  # Give up after 50 lines
                start_pos = 0
                break
            if line.strip() and not any(keyword in line.lower() for keyword in 
                ['project gutenberg', 'ebook', 'www.gutenberg', 'license', 'trademark']):
                start_pos = text.find(line)
                break
    
    content = text[start_pos:end_pos] if start_pos != -1 else text[:end_pos]
    
    # Remove common non-content patterns
    # Remove illustration markers
    content = re.sub(r'\[Illustration[^\]]*\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[Illustration\]', '', content, flags=re.IGNORECASE)
    
    # Remove page markers
    content = re.sub(r'\[Sidenote:.*?\]', '', content, flags=re.DOTALL)
    content = re.sub(r'\[Footnote.*?\]', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove transcriber's notes
    content = re.sub(r'Transcriber[\'s]* Note[s]*:.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove table of contents (common patterns)
    # Match "Contents" or "CONTENTS" followed by chapter listings
    content = re.sub(
        r'\n\s*(?:Contents|CONTENTS|Table of Contents|TABLE OF CONTENTS)\s*\n(?:.*?\n){1,50}?(?=\n\n[A-Z]|\nCHAPTER)',
        '\n',
        content,
        flags=re.IGNORECASE
    )
    
    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n\s*\n\s*\n(\s*\n)+', '\n\n\n', content)
    
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # Remove very short lines at the beginning that might be formatting artifacts
    lines = content.split('\n')
    while lines and len(lines[0].strip()) < 3:
        lines.pop(0)
    content = '\n'.join(lines)
    
    return content


def clean_dataset(input_dir: Path, output_dir: Path, min_length: int = 1000) -> None:
    """Clean all text files in the input directory.
    
    Args:
        input_dir: Directory containing raw Gutenberg texts
        output_dir: Directory to save cleaned texts
        min_length: Minimum character length for a valid cleaned text
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {input_dir}")
    
    cleaned_count = 0
    skipped_count = 0
    failed_count = 0
    
    for txt_file in txt_files:
        try:
            # Read the raw text
            raw_text = txt_file.read_text(encoding='utf-8', errors='ignore')
            
            # Clean the text
            cleaned_text = clean_gutenberg_text(raw_text)
            
            # Validate the cleaned text
            if len(cleaned_text) < min_length:
                print(f"Skipped {txt_file.name}: too short after cleaning ({len(cleaned_text)} chars)")
                skipped_count += 1
                continue
            
            # Save the cleaned text
            output_file = output_dir / txt_file.name
            output_file.write_text(cleaned_text, encoding='utf-8')
            
            cleaned_count += 1
            if cleaned_count % 100 == 0:
                print(f"Processed {cleaned_count} files...")
                
        except Exception as e:
            print(f"Failed to process {txt_file.name}: {e}")
            failed_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Cleaning complete!")
    print(f"  Successfully cleaned: {cleaned_count}")
    print(f"  Skipped (too short): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Project Gutenberg texts for LLM training")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset",
        help="Directory containing raw Gutenberg text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cleaned_dataset",
        help="Directory to save cleaned text files"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum character length for valid cleaned text (default: 1000)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    clean_dataset(input_dir, output_dir, args.min_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
