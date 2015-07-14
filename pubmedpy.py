#!/usr/bin/env python
# encoding: utf-8
#
# 
# The MIT License
# 
# Copyright (c) 2009 Byron C Wallace
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# http://www.opensource.org/licenses/mit-license.php

"""
Byron C Wallace
pubmedpy.py
--

Example use:

>python pubmed_fetchr.py -e byron.wallace@gmail.com -s biopython

(NCBI wants your email address for the web services stuff). -s is the search string, here we search 
for abstracts related to "biopython". 

"""

# std libraries
import sys
import getopt
import pdb
import os
from optparse import OptionParser
import Bio
from Bio import Entrez
from Bio import Medline
import nltk

import operator 

Entrez.email = "byron.wallace@gmail.com"

# home-grown
#import tfidf2

def main(argv=None):
  parser = OptionParser()
  parser.add_option("-e", "--email", action="store", type="string", dest="email")
  parser.add_option("-s", "--search_string", action="store", type="string", dest="search_str")
  (options, args) = parser.parse_args()
  
  Entrez.email = options.email
  records = article_search("pubmed", options.search_str)
  print "Writing out title and abstract data..."
  write_out_fields(["TI", "AB"], records)
  print("fin.")
    
def set_email(email):
    Entrez.email = email


def fetch_and_encode(article_ids, out_dir, binary_features=False, 
                                    labels=None, fields = ["AB", "TI"], out_f_name = ""):
    '''
    First fetches from the web, then encodes them.
    '''
    # first, fetch the articles
    fetch_and_write_out(article_ids, out_dir, fields = fields)
    
    for field in fields:   
        print "encoding %s..." %field
        # now, clean and encode them
        out_for_field = os.path.join(out_dir, field)
        tfidf2.encode_docs(out_for_field, os.path.join(out_for_field, "encoded"), out_f_name + field, lbl_dict = labels)
    print "finito."
    

def fetch_articles(article_ids):
    #print "fetching abstracts..."
    print article_ids
    handle = Entrez.efetch(db="pubmed",id=article_ids,rettype="medline",retmode="text")
    
    #dbids = EUtils.DBIds("pubmed", [str(xid) for xid in article_ids])
    #client = ThinClient.ThinClient()

    # fix for new NCBI API
    #handle = client.efetch_using_dbids(dbids=dbids, rettype='medline', retmode='text')
    #handle = EUtils.efetch(db="pubmed",id=article_ids,rettype="medline",retmode="text")
    records = Medline.parse(handle)
    print "Done." 
    return records   
    
def batch_fetch(article_ids, batch_size=20):
    all_records = []
    total = len(article_ids)
    fetched_so_far = 0
    while fetched_so_far < total:
      records = fetch_articles(article_ids[fetched_so_far:fetched_so_far+batch_size])
      fetched_so_far += batch_size
      all_records.extend([r for r in records])
      print "fetched so far, total to fetch: %s, %s" % (fetched_so_far, total)
  
    return all_records
  

def article_search(db, search_str):
    handle = Entrez.esearch(db=db,term=search_str,retmax=1000)
    record = Entrez.read(handle)
    article_ids = record["IdList"]
    print "Found %s articles" % len(article_ids)
    return batch_fetch(article_ids)
  
 
def write_to_tsv(field_keys, records, fpath):
    header_line = "\t".join(field_keys)
    all_lines = [header_line]
    for r in records:
        cur_line = []
        for field in field_keys:
            cur_line.append(str(r[field]))
        all_lines.append("\t".join(cur_line))
    outf = open(fpath, 'w')
    outf.write("\n".join(all_lines))
    
def write_out_fields(field_keys, records, base_out_dir):
    # make a directory for each field
    if not os.path.isdir(base_out_dir):
        os.mkdir(base_out_dir)
        
    for field in field_keys:
      field_dir = os.path.join(base_out_dir, field)
      if not os.path.isdir(field_dir):
        os.mkdir(field_dir)
    # write out 
    for record in records:
      for field in field_keys:
        out_f = open(os.path.join(base_out_dir, field, record["PMID"]), "w")
        cur_field = None
        try:
          cur_field = record[field]
        except Exception, inst:
          cur_field = ""
          
        if isinstance(cur_field, list):
            cur_field = ", ".join(cur_field)
        out_f.write(cur_field)
        out_f.close()


def get_pmid_from_bib_info(journal, volume, issue, pages):
  # e.g.
  # JAMA Surg.[Journal] 3[Issue] 148[Volume] 259-263[Pagination]
  search_str = "%s[Journal] %s[Volume] %s[Issue] %s[Pagination]" % (
                  journal, volume, issue, pages)
  handle = Entrez.esearch(db="pubmed", term="%s" % search_str)
  records = Entrez.read(handle)
  id_list = records["IdList"]
  #pdb.set_trace()
  #pdb.set_trace()
  if len(id_list) == 0:
    # we failed, I guess
    print "FAILURE for %s" % search_str
    return False
  elif len(id_list) > 1:
    pdb.set_trace()
  return id_list[0]

def get_pmid_from_doi(doi_str):
    handle = Entrez.esearch(db="pubmed",term='"{0}"'.format(doi_str))
    # what did we get?
    records = Entrez.read(handle)
    id_list = records["IdList"]
    #pdb.set_trace()
    if len(id_list) == 0:
      # exact match not found; drop the quotes (back-off)
      try:
        handle = Entrez.esearch(db="pubmed",term='{0}'.format(title))
        records = Entrez.read(handle)
        id_list = records["IdList"]
      except:
        return False
        
    # one? more than one?
    num_records = len(id_list)
    print "%s records found for %s" % (num_records, doi_str)
    if num_records == 0:
        print "no records found for '{0}'".format(title)
        return False

    return id_list[0]

def get_pmid_from_title(title, DISTANCE_THRESHOLD=7):
    '''
    searches PubMed title field for the given
    text. if a match is found with an edit
    distance of DISTANCE_THRESHOLD, return its ID;
    otherwise return False, indicating that no 
    match was found.
    '''

    # first try to find the literal match
    handle = Entrez.esearch(db="pubmed",term = title,field="ti")
    # what did we get?
    records = Entrez.read(handle)
    id_list = records["IdList"]
    #pdb.set_trace()
    if len(id_list) == 0:
      # exact match not found; drop the quotes (back-off)
      try:
        handle = Entrez.esearch(db="pubmed",term='{0}'.format(title))
        records = Entrez.read(handle)
        id_list = records["IdList"]
      except:
        return False
        
    # one? more than one?
    num_records = len(id_list)
    if num_records == 0:
        print "no records found for '{0}'".format(title)
        return False
    

    # if we get more > 1 record(s), sort them by proximity
    # to the search string, w.r.t. edit distance.
    # do this even if we only get one record, because we
    # get the edit distance this way
    if title.endswith("."):
      title = title[:-1]
    sorted_pmids = rank_by_edit_distance(title.strip().lower(), id_list)
    if sorted_pmids is None:
      return False

    best_pmid, best_distance = sorted_pmids[0]
    

    if best_distance <= DISTANCE_THRESHOLD:
      print "success!"
      return best_pmid
    print "no dice -- closest match ({0}) had edit distance of {1}".format(best_pmid, best_distance)
    return False # fail

def rank_by_edit_distance(target_str, pmids):
  ''' 
  ranks the ids that are the keys in ids_to_strs by
  their proximity to target_str, with respect to edit
  distance
  '''
  scores = {}
  for pmid in pmids:
    retrieved = list(fetch_articles([pmid]))
    if len(retrieved) == 0:
      break

    title = None
    try:
      title = retrieved[0]["TI"].lower() # assuming this record/pmid exists!
    except:
      print "what the.. ? no TI field for PMID {0}".format(pmid)
      break

    #pdb.set_trace()
    scores[pmid] = nltk.metrics.distance.edit_distance(target_str, title)

  if len(scores) < 1:
    return None
    
  sorted_pmids = sorted(scores.iteritems(), key=operator.itemgetter(1))
  return sorted_pmids


if __name__ == "__main__":
  sys.exit(main())
