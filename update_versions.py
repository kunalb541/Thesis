#!/usr/bin/env python3
"""
Version Update Script
=====================
Updates all version numbers to a consistent version across the codebase.

Usage:
    python update_versions.py --version 10.0 [--dry-run]
"""

import argparse
import re
from pathlib import Path
from datetime import datetime

# Files to update with their version patterns
VERSION_FILES = {
    'README.md': [
        (r'### Version \d+\.\d+ \(Current\)', '### Version {version} (Current)'),
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'code/config.py': [
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'code/evaluate.py': [
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'code/simulate.py': [
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'code/train.py': [
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'code/transformer.py': [
        (r'Version: \d+\.\d+', 'Version: {version}'),
    ],
    'docs/RESEARCH_GUIDE.md': [
        (r'\*\*Version \d+\.\d+', '**Version {version}'),
    ],
}


def update_version_in_file(filepath, patterns, new_version, dry_run=False):
    """Update version in a single file"""
    path = Path(filepath)
    
    if not path.exists():
        print(f"⚠️  File not found: {filepath}")
        return False
    
    content = path.read_text()
    original_content = content
    changes_made = False
    
    for pattern, replacement in patterns:
        replacement_str = replacement.format(version=new_version)
        new_content, count = re.subn(pattern, replacement_str, content)
        
        if count > 0:
            content = new_content
            changes_made = True
            print(f"  ✓ Updated {count} occurrence(s) of '{pattern}'")
    
    if changes_made:
        if not dry_run:
            path.write_text(content)
            print(f"✅ Updated: {filepath}")
        else:
            print(f"🔍 Would update: {filepath}")
        return True
    else:
        print(f"  No changes needed: {filepath}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Update version numbers across codebase')
    parser.add_argument('--version', type=str, required=True,
                       help='New version number (e.g., 10.0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    # Validate version format
    if not re.match(r'^\d+\.\d+$', args.version):
        print("❌ Error: Version must be in format X.Y (e.g., 10.0)")
        return
    
    print("="*70)
    print(f"VERSION UPDATE SCRIPT")
    print("="*70)
    print(f"Target version: {args.version}")
    print(f"Dry run: {args.dry_run}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    total_updated = 0
    
    for filepath, patterns in VERSION_FILES.items():
        print(f"\n📄 Processing: {filepath}")
        if update_version_in_file(filepath, patterns, args.version, args.dry_run):
            total_updated += 1
    
    print("\n" + "="*70)
    if args.dry_run:
        print(f"🔍 DRY RUN: Would update {total_updated} files")
        print("Run without --dry-run to apply changes")
    else:
        print(f"✅ SUCCESS: Updated {total_updated} files to version {args.version}")
    print("="*70)


if __name__ == '__main__':
    main()