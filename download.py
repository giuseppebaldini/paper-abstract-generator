#!/usr/bin/env python
# coding: utf-8

import requests
import shutil
import gzip

# download zipped anthology file from ACL website
url = 'https://aclanthology.org/anthology+abstracts.bib.gz'
r = requests.get(url, allow_redirects=True)
in_filepath = 'data/anthology+abstracts.bib.gz'

with open(in_filepath, 'wb') as f:
	write_data = f.write(r.content)
	
	# unzip anthology file and save in data folder
	out_filepath = 'data/anthology+abstracts.bib'
	with gzip.open(in_filepath, 'rb') as f_in:
		with open(out_filepath, 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
