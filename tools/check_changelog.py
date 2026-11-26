#!/usr/bin/env python3
"""Require a dated README changelog entry whenever tracked project files change."""
import argparse,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HEADING='## 更新日志'

def section(text):
    if HEADING not in text:raise ValueError('README must include 更新日志')
    return text.split(HEADING,1)[1].split('\n## ',1)[0]

def entries(text):return set(re.findall(r'^- \d{4}-\d{2}-\d{2}：.+$',section(text),re.M))

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,encoding='utf-8').strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base');args=parser.parse_args();readme=(ROOT/'README.md').read_text(encoding='utf-8')
    if not entries(readme):raise ValueError('Add a brief dated changelog item to README')
    if args.base and set(args.base)!={'0'}:
        changed=git('diff','--name-only',args.base).splitlines()
        if any(name!='README.md' for name in changed):
            try:before=git('show',args.base+':README.md')
            except subprocess.CalledProcessError:before=HEADING+'\n'
            if not entries(readme)-entries(before):raise ValueError('Project files changed: add a new dated item under README 更新日志')
    print('README changelog check passed')

if __name__=='__main__':main()
