{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8ffab04c-7aaa-4f73-a089-fe32d423c166",
   "metadata": {},
   "source": [
    "# Data input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4aaf26c7-a7ad-4c1a-8908-d9e42f72afc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "import shutil\n",
    "import gzip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8d6ce001-3181-431b-b42d-8476f7b7d1f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# download zipped anthology file from ACL website\n",
    "url = 'https://aclanthology.org/anthology+abstracts.bib.gz'\n",
    "r = requests.get(url, allow_redirects=True)\n",
    "in_filepath = 'data/anthology+abstracts.bib.gz'\n",
    "\n",
    "with open(in_filepath, 'wb') as f:\n",
    "    write_data = f.write(r.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7773ebaa-c11e-4b7f-ae4f-6a8f7ed859e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# unzip anthology file and save in data folder\n",
    "out_filepath = 'data/anthology+abstracts.bib'\n",
    "\n",
    "with gzip.open(in_filepath, 'rb') as f_in:\n",
    "    with open(out_filepath, 'wb') as f_out:\n",
    "        shutil.copyfileobj(f_in, f_out)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f87bdd41-c1da-479f-a445-a71c5d6f6f2b",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Parse bibtex files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "758e2af7-5eec-40a1-9a50-b47d1c6d98a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "bib_data = parse_file('data/anthology+abstracts.bib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d45a821-4187-4fd1-958f-156a4ce67b30",
   "metadata": {},
   "outputs": [],
   "source": [
    "list(bib_data.entries.keys())[-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fdacbf05-d477-4890-a075-12cf4bff29be",
   "metadata": {},
   "outputs": [],
   "source": [
    "len(list(bib_data.entries.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db084bae-b3a7-40cd-a085-6d29ddfff580",
   "metadata": {},
   "outputs": [],
   "source": [
    "bib_data.entries['lieberman-etal-1965-automatic'].fields['year']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a3fd3dd-9aee-4a0d-a5c8-4df8bdcdf384",
   "metadata": {},
   "outputs": [],
   "source": [
    "for k in bib_data.entries.keys():\n",
    "    try:\n",
    "        year = bib_data.entries[k].fields['year']\n",
    "        abstract = bib_data.entries[k].fields['abstract']\n",
    "        \n",
    "        if year > '2015':\n",
    "            f = open('data/datasets/abstracts_%s.txt' %year, 'a')\n",
    "            f.write(abstract)\n",
    "            f.close()\n",
    "            \n",
    "    except (KeyError, UnicodeEncodeError): # entries without abstracts are excluded\n",
    "        pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41e20722-8bc0-443b-8ec8-04acc6d3dc49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# eliminate stop words\n",
    "def tokenize_input(input):\n",
    "    # make everything lowercase\n",
    "    input = input.lower()\n",
    "\n",
    "    # use tokenizer\n",
    "    tokenizer = RegexpTokenizer(r'\\w+')\n",
    "    tokens = tokenizer.tokenize(input)\n",
    "\n",
    "    # filter out stopwords\n",
    "    final = filter(lambda token: token not in stopwords.words('english'), tokens)\n",
    "    \n",
    "    # end result in final\n",
    "    return \" \".join(final)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5be362e8-c8a9-45e3-9482-2ce95a56c410",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for year in range(2016,2022):        \n",
    "        with open('data/datasets/abstracts_%s.txt' %year) as abstr:\n",
    "            lines = abstr.readlines()\n",
    "            processed = tokenize_input(lines[0])\n",
    "            \n",
    "            # create individual year files\n",
    "            y = open('data/datasets/%s.txt' %year, 'a')\n",
    "            y.write(processed)\n",
    "            \n",
    "            # create all years file\n",
    "            a = open('data/datasets/all.txt', 'a')\n",
    "            a.write(processed)\n",
    "            \n",
    "            y.close()\n",
    "            a.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6dc1ede8-dbb0-4428-929a-33d9d64278b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "for k in bib_data.entries.keys():\n",
    "    try:\n",
    "        year = bib_data.entries[k].fields['year']\n",
    "        abstract = bib_data.entries[k].fields['abstract']\n",
    "        \n",
    "        if year > '2015':\n",
    "            f = open('data/datasets/abstracts_%s.txt' %year, 'a')\n",
    "            f.write(abstract + '\\n')\n",
    "            f.close()\n",
    "            \n",
    "    except (KeyError, UnicodeEncodeError): # entries without abstracts are excluded\n",
    "        pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1236,
   "id": "836fe161-e766-4ec4-ba37-89f6b3a5b8a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "f = open('data/datasets/abstracts_2020.txt')\n",
    "text = f.read()\n",
    "abstracts_2020 = text.split('\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1237,
   "id": "2f759ece-f8e4-4e96-bcee-24162302f708",
   "metadata": {},
   "outputs": [],
   "source": [
    "trimmed = [remove_stopwords(a) for a in abstracts_2020]\n",
    "lowercase = [a.lower() for a in trimmed]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1238,
   "id": "108aadc8-7973-4e89-9614-7027bdaa7a79",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = RegexpTokenizer(r'\\w+')\n",
    "tokenized = [tokenizer.tokenize(a) for a in lowercase]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1239,
   "id": "5fdb67ee-708c-49bb-96ad-b2243598fbf5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6499"
      ]
     },
     "execution_count": 1239,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# total number of tokenized abstracts\n",
    "len(tokenized)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1241,
   "id": "0da3e223-a7fd-4c1f-93eb-9164d470af7e",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[['the',\n",
       "  'relatedness',\n",
       "  'research',\n",
       "  'articles',\n",
       "  'patents',\n",
       "  'court',\n",
       "  'rulings',\n",
       "  'web',\n",
       "  'pages',\n",
       "  'document',\n",
       "  'types',\n",
       "  'calculated',\n",
       "  'citation',\n",
       "  'hyperlink',\n",
       "  'based',\n",
       "  'approaches',\n",
       "  'like',\n",
       "  'co',\n",
       "  'citation',\n",
       "  'proximity',\n",
       "  'analysis',\n",
       "  'the',\n",
       "  'main',\n",
       "  'limitation',\n",
       "  'citation',\n",
       "  'based',\n",
       "  'approaches',\n",
       "  'documents',\n",
       "  'receive',\n",
       "  'little',\n",
       "  'citations',\n",
       "  'we',\n",
       "  'propose',\n",
       "  'virtual',\n",
       "  'citation',\n",
       "  'proximity',\n",
       "  'vcp',\n",
       "  'siamese',\n",
       "  'neural',\n",
       "  'network',\n",
       "  'architecture',\n",
       "  'combines',\n",
       "  'advantages',\n",
       "  'co',\n",
       "  'citation',\n",
       "  'proximity',\n",
       "  'analysis',\n",
       "  'diverse',\n",
       "  'notions',\n",
       "  'relatedness',\n",
       "  'high',\n",
       "  'recommendation',\n",
       "  'performance',\n",
       "  'advantage',\n",
       "  'content',\n",
       "  'based',\n",
       "  'filtering',\n",
       "  'high',\n",
       "  'coverage',\n",
       "  'vcp',\n",
       "  'trained',\n",
       "  'corpus',\n",
       "  'documents',\n",
       "  'textual',\n",
       "  'features',\n",
       "  'real',\n",
       "  'citation',\n",
       "  'proximity',\n",
       "  'ground',\n",
       "  'truth',\n",
       "  'vcp',\n",
       "  'predicts',\n",
       "  'documents',\n",
       "  'based',\n",
       "  'title',\n",
       "  'abstract',\n",
       "  'proximity',\n",
       "  'documents',\n",
       "  'co',\n",
       "  'cited',\n",
       "  'co',\n",
       "  'cited',\n",
       "  'the',\n",
       "  'prediction',\n",
       "  'way',\n",
       "  'real',\n",
       "  'citation',\n",
       "  'proximity',\n",
       "  'calculate',\n",
       "  'document',\n",
       "  'relatedness',\n",
       "  'uncited',\n",
       "  'documents',\n",
       "  'in',\n",
       "  'evaluation',\n",
       "  '2',\n",
       "  'million',\n",
       "  'co',\n",
       "  'citations',\n",
       "  'wikipedia',\n",
       "  'articles',\n",
       "  'vcp',\n",
       "  'achieves',\n",
       "  'mae',\n",
       "  '0',\n",
       "  '0055',\n",
       "  'i',\n",
       "  'e',\n",
       "  'improvement',\n",
       "  '20',\n",
       "  'baseline',\n",
       "  'learning',\n",
       "  'curve',\n",
       "  'suggests',\n",
       "  'work',\n",
       "  'needed'],\n",
       " ['the',\n",
       "  'question',\n",
       "  'utility',\n",
       "  'blind',\n",
       "  'peer',\n",
       "  'review',\n",
       "  'fundamental',\n",
       "  'scientific',\n",
       "  'research',\n",
       "  'some',\n",
       "  'studies',\n",
       "  'investigate',\n",
       "  'exactly',\n",
       "  'blind',\n",
       "  'papers',\n",
       "  'double',\n",
       "  'blind',\n",
       "  'review',\n",
       "  'manually',\n",
       "  'automatically',\n",
       "  'identifying',\n",
       "  'true',\n",
       "  'authors',\n",
       "  'mainly',\n",
       "  'suggesting',\n",
       "  'number',\n",
       "  'self',\n",
       "  'citations',\n",
       "  'submitted',\n",
       "  'manuscripts',\n",
       "  'primary',\n",
       "  'signal',\n",
       "  'identity',\n",
       "  'however',\n",
       "  'related',\n",
       "  'work',\n",
       "  'automated',\n",
       "  'approaches',\n",
       "  'limited',\n",
       "  'sizes',\n",
       "  'datasets',\n",
       "  'restricted',\n",
       "  'experimental',\n",
       "  'setup',\n",
       "  'lack',\n",
       "  'practical',\n",
       "  'insights',\n",
       "  'blind',\n",
       "  'review',\n",
       "  'process',\n",
       "  'in',\n",
       "  'work',\n",
       "  'train',\n",
       "  'models',\n",
       "  'identify',\n",
       "  'authors',\n",
       "  'affiliations',\n",
       "  'nationalities',\n",
       "  'real',\n",
       "  'world',\n",
       "  'large',\n",
       "  'scale',\n",
       "  'experiments',\n",
       "  'microsoft',\n",
       "  'academic',\n",
       "  'graph',\n",
       "  'including',\n",
       "  'cold',\n",
       "  'start',\n",
       "  'scenario',\n",
       "  'our',\n",
       "  'models',\n",
       "  'accurate',\n",
       "  'identify',\n",
       "  'authors',\n",
       "  'affiliations',\n",
       "  'nationalities',\n",
       "  'held',\n",
       "  'out',\n",
       "  'papers',\n",
       "  '40',\n",
       "  '3',\n",
       "  '47',\n",
       "  '9',\n",
       "  '86',\n",
       "  '0',\n",
       "  'accuracy',\n",
       "  'respectively',\n",
       "  'top',\n",
       "  '10',\n",
       "  'guesses',\n",
       "  'models',\n",
       "  'however',\n",
       "  'insights',\n",
       "  'model',\n",
       "  'demonstrate',\n",
       "  'entities',\n",
       "  'identifiable',\n",
       "  'small',\n",
       "  'number',\n",
       "  'guesses',\n",
       "  'primarily',\n",
       "  'combination',\n",
       "  'self',\n",
       "  'citations',\n",
       "  'social',\n",
       "  'common',\n",
       "  'citations',\n",
       "  'moreover',\n",
       "  'analysis',\n",
       "  'results',\n",
       "  'leads',\n",
       "  'interesting',\n",
       "  'findings',\n",
       "  'prominent',\n",
       "  'affiliations',\n",
       "  'easily',\n",
       "  'identifiable',\n",
       "  'e',\n",
       "  'g',\n",
       "  '93',\n",
       "  '8',\n",
       "  'test',\n",
       "  'papers',\n",
       "  'written',\n",
       "  'microsoft',\n",
       "  'identified',\n",
       "  'top',\n",
       "  '10',\n",
       "  'guesses',\n",
       "  'the',\n",
       "  'experimental',\n",
       "  'results',\n",
       "  'show',\n",
       "  'conventional',\n",
       "  'belief',\n",
       "  'self',\n",
       "  'citations',\n",
       "  'informative',\n",
       "  'looking',\n",
       "  'common',\n",
       "  'citations',\n",
       "  'suggesting',\n",
       "  'removing',\n",
       "  'self',\n",
       "  'citations',\n",
       "  'sufficient',\n",
       "  'authors',\n",
       "  'maintain',\n",
       "  'anonymity']]"
      ]
     },
     "execution_count": 1241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# example of tokenized abstract\n",
    "tokenized[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 397,
   "id": "54719972-0e1d-4779-881e-02d6e7179dbb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'patents'"
      ]
     },
     "execution_count": 397,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# single word\n",
    "tokenized[0][4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 398,
   "id": "6e869795-650b-4da4-827d-3f870c58e061",
   "metadata": {},
   "outputs": [],
   "source": [
    "words = [word for abstract in tokenized for word in abstract] # could use itertools to improve performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 399,
   "id": "9008fc62-c3e3-4443-a43f-43f72103a848",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "606420"
      ]
     },
     "execution_count": 399,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(words)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
