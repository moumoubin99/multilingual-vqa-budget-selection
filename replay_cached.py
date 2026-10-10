#!/usr/bin/env python3
"""CPU statistical replay of preserved predictions; never import/load a model."""
from __future__ import annotations
import argparse, ast, collections, gzip, hashlib, json, math, statistics, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def readrows(p):
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def need(ok, message):
    if not ok:
        raise AssertionError(message)

def compare(expected, actual, label, report, ignore=frozenset(), tol=5e-12):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(expected)-ignore == set(actual)-ignore, label+' keys')
        for k in sorted(set(expected)-ignore):
            compare(expected[k], actual[k], label+'/'+k, report, ignore, tol)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(expected) == len(actual), label+' length')
        for i, (a,b) in enumerate(zip(expected, actual)):
            compare(a,b,label+'/'+str(i),report,ignore,tol)
    elif isinstance(expected, (float,int)) and not isinstance(expected,bool):
        need(isinstance(actual,(float,int)) and math.isfinite(actual) and abs(expected-actual)<=tol,
             label+' numeric mismatch')
        report['numeric_comparisons'] += 1
    else:
        need(expected == actual, label+' value mismatch')

def pure_functions(file, names, ns):
    tree = ast.parse(Path(file).read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in functions} == set(names), 'missing source functions')
    unit = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(file),'exec'),ns)

def trim(values):
    ordered = sorted(values)
    t = math.ceil(len(ordered)/4)
    return statistics.mean(ordered[t:len(ordered)-t])

def exact_p(pos,neg):
    n=pos+neg
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n)

def table3(contrast, ns):
    out={}
    for name,test in contrast['tests'].items():
        v=test['beneficial_positive_values'];pos=sum(x>1e-12 for x in v);neg=sum(x<-1e-12 for x in v)
        out[name]={'n':len(v),'median':statistics.median(v),'positive':pos,'negative':neg,
                   'ties':len(v)-pos-neg,'p_two_sided_ties_excluded':exact_p(pos,neg),
                   'order_statistic_interval_alpha_0.005':ns['order_statistic_interval'](v,0.005)}
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,required=True,help='directory outside this repository')
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    need(not out.is_relative_to(root),'output must be outside the immutable repository')
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    report={'status':'RUNNING','no_training':True,'no_model_loading':True,'no_fresh_inference':True,
            'numeric_comparisons':0,'limitations':['Transfer remains 27/73 complete; no missing outputs are synthesized.',
            'The original full shortcut probability rows were overwritten. Table 4 shortcut metrics cannot be completely replayed.',
            'Metadata and summaries are documented derivatives; original byte hashes do not identify the derived files.',
            'No claim of fresh inference, cross-runtime numerical identity, or confirmatory success.']}
    try:
        manifest=readj(root/'MANIFEST.json')
        for entry in manifest['files']:
            p=root/entry['path'];need(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'manifest '+entry['path'])
        report['manifest_files_verified']=len(manifest['files'])
        originals=readj(root/'provenance/original_sources.json')['files']
        original_by_release={x['release_path']:x for x in originals}
        exact_sources=0
        for entry in originals:
            if entry['transformation']=='Exact source bytes':
                need(sha(root/entry['release_path'])==entry['source_sha256'],'original execution source fingerprint')
                exact_sources+=1
        report['exact_execution_sources_verified']=exact_sources
        ns={'np':np,'math':math,'defaultdict':collections.defaultdict}
        pure_functions(root/'code/src/ccmvqa/formulas.py',['selective_risk'],ns)
        pure_functions(root/'code/scripts/evaluate_vqa.py',['language_matrices','unweighted_accuracy','confirmatory_secondary_metrics'],ns)
        pure_functions(root/'code/scripts/analyze_registered_tests.py',['binomial_cdf','order_statistic_interval'],ns)
        pool=readrows(root/'data/pool_metadata.jsonl.gz');ep=readrows(root/'data/endpoint_metadata.jsonl.gz');cv=readrows(root/'data/cvqa_metadata.jsonl.gz')
        need((len(pool),len(ep),len(cv))==(29999,3600,1800),'metadata counts')
        pm={r['id']:r for r in pool};need(len(pm)==29999,'pool unique ids')
        need(not {r['concept_id'] for r in pool}&{r['concept_id'] for r in ep},'pool/endpoint concept overlap')
        reg=readj(root/'data/subset_registry.json');need(len(reg['subsets'])==162,'registry count');sigs=[]
        for e in reg['subsets']:
            subset=readj(root/e['release_path']);ids=subset['ordered_ids'];need(len(ids)==len(set(ids))==2000,'subset unique ids')
            sig=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest();need(sig==e['row_id_signature'],'subset signature');sigs.append(sig)
            need(dict(collections.Counter(pm[i]['language'] for i in ids))==e['language_marginal'],'subset language marginal')
            tr=readj(root/'training'/(e['arm'].lower()+'-s'+str(e['seed'])+'.json'))
            need(tr['input_sha256']==subset['original_file_sha256'] and tr['updates']==63 and tr['examples']==2000,'training source binding')
        need(len(set(sigs))==162,'subset distinct signatures')
        report['datasets']={'pool':len(pool),'endpoint':len(ep),'transfer':len(cv),'subsets':162,'trained_runs':162,
                            'distinct_signatures':162,'pool_endpoint_concept_overlap':0,'updates_per_run':63}
        metrics={};recomputed_summaries={}
        for scope,endpoint,expected in [('worldcuisines',ep,163),('cvqa',cv,27)]:
            paths=sorted((root/'predictions'/scope).glob('*.jsonl.gz'));need(len(paths)==expected,'prediction file count')
            metrics[scope]={};recomputed_summaries[scope]={}
            for p in paths:
                run=p.name.removesuffix('.jsonl.gz');rs=readrows(p);summary=readj(root/'summaries'/scope/(run+'.json'))
                source_entry=original_by_release[p.relative_to(root).as_posix()]
                with gzip.open(p,'rb') as f:original_prediction_bytes=f.read()
                need(len(original_prediction_bytes)==source_entry['source_bytes'] and hashlib.sha256(original_prediction_bytes).hexdigest()==source_entry['source_sha256'],'lossless original prediction fingerprint')
                need(summary['status']=='completed' and summary['full_generation'] is True and len(rs)==len(endpoint),'completed generation evaluation')
                for i,(r,source) in enumerate(zip(rs,endpoint)):
                    prob=np.asarray(r['semantic_probability'],dtype=np.float64)
                    need(r['id']==source['id'] and r['language']==source['language'] and r['frozen_index']==i,'frozen prediction axis')
                    need(prob.shape==(source['n_options'],) and np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and abs(prob.sum()-1)<5e-12,'probability schema')
                    pred=int(prob.argmax());need(r['gold_index']==source['gold_index'] and r['prediction_index']==pred and r['correct']==int(pred==r['gold_index']) and abs(r['confidence']-prob[pred])<5e-12,'prediction identity')
                    need(all(sorted(x)==list(range(len(prob))) for x in r['permutations']) and r['permutations'][1]==r['permutations'][0][::-1],'complementary permutations')
                langs,correct,confidence,mask,ids=ns['language_matrices'](rs)
                risks,accepted,thresholds=ns['selective_risk'](correct,confidence,mask,ids,summary['target_coverage'])
                observed={'accuracy':ns['unweighted_accuracy'](rs),**ns['confirmatory_secondary_metrics'](rs),
                    'per_language_selective_risk':dict(zip(langs,risks.tolist())),
                    'per_language_threshold_report_only':dict(zip(langs,thresholds.tolist())),
                    'accepted_counts':dict(zip(langs,accepted.sum(axis=1).astype(int).tolist()))}
                compare({k:summary[k] for k in observed},observed,scope+'/'+run,report)
                perlang={lang:statistics.mean(r['correct'] for r in rs if r['language']==lang) for lang in langs}
                metrics[scope][run]={'accuracy':observed['accuracy'],'trimmed_accuracy':trim(list(perlang.values())),'per_language_accuracy':perlang,
                                     'aurc':observed['aurc_secondary'],'brier':observed['brier_secondary']}
                recomputed_summaries[scope][run]={**summary,**observed}
        wc=metrics['worldcuisines'];seeds=sorted({e['seed'] for e in reg['subsets']});arms=sorted({e['arm'] for e in reg['subsets']})
        table2={'base':wc['base-noadapter'],'policies':{}}
        for arm in arms:
            rs=[wc[arm.lower()+'-s'+str(s)] for s in seeds];values=[r['trimmed_accuracy'] for r in rs]
            table2['policies'][arm]={'n':len(values),'trimmed_accuracy_median':statistics.median(values),'trimmed_accuracy_sd':statistics.stdev(values),
                'unweighted_accuracy_median':statistics.median(r['accuracy'] for r in rs),'trimmed_accuracy_by_seed':dict(zip(map(str,seeds),values))}
        contrast=readj(root/'analysis/CONTRAST_TABLE.json');rebuilt=dict(contrast)
        sr=recomputed_summaries['worldcuisines'];langs=contrast['endpoint']['languages']
        tests={}
        for name,comp in [('T01','Random'),('T02','LangOnlyCond'),('T03','LangOnly'),('T04','ConceptOnly'),('T05','EnglishHeavy')]:
            per={str(s):{l:sr[comp.lower()+'-s'+str(s)]['per_language_selective_risk'][l]-sr['joint-s'+str(s)]['per_language_selective_risk'][l] for l in langs} for s in seeds}
            tests[name]={**contrast['tests'][name],'per_language_contrast':per,'beneficial_positive_values':[trim(list(per[str(s)].values())) for s in seeds]}
        per={str(s):{l:-(sr['joint-s'+str(s)]['per_language_selective_risk'][l]-sr['langonly-s'+str(s)]['per_language_selective_risk'][l]-sr['conceptonly-s'+str(s)]['per_language_selective_risk'][l]+sr['natnat-s'+str(s)]['per_language_selective_risk'][l]) for l in langs} for s in seeds}
        tests['T07']={**contrast['tests']['T07'],'per_language_contrast':per,'beneficial_positive_values':[trim(list(per[str(s)].values())) for s in seeds]}
        rebuilt['tests']=tests
        rebuilt['descriptive_per_language_accuracy']={run:wc[run]['per_language_accuracy'] for run in contrast['descriptive_per_language_accuracy']}
        occ=dict(contrast['descriptive_only']['arm_level_occupancy'])
        occ['values']=[trim([sr['lowocc-s'+str(s)]['per_language_selective_risk'][l]-sr['highocc-s'+str(s)]['per_language_selective_risk'][l] for l in langs]) for s in seeds]
        rebuilt['descriptive_only']={'arm_level_occupancy':occ};compare(contrast,rebuilt,'CONTRAST_TABLE',report)
        t3=table3(rebuilt,ns);ties=readj(root/'analysis/SIGN_COUNTS_TIES.json')['tests']
        for name,t in t3.items():compare(ties[name],{k:t[k] for k in ties[name]},'corrected_ties/'+name,report)
        transfer=metrics['cvqa'];ref=readj(root/'analysis/TRANSFER_PARTIAL.json');cvout={'base':{'acc':transfer['base-noadapter']['accuracy'],'trimmed':transfer['base-noadapter']['trimmed_accuracy'],'per_lang':transfer['base-noadapter']['per_language_accuracy']}}
        for arm in ['random','joint']:
            ss=sorted(run.split('-s')[1] for run in transfer if run.startswith(arm+'-s'));rs=[transfer[arm+'-s'+s] for s in ss]
            cvout[arm]={'n_seeds':len(ss),'acc_median':statistics.median(r['accuracy'] for r in rs),'trimmed_median':statistics.median(r['trimmed_accuracy'] for r in rs),
                'acc_sd':statistics.stdev(r['accuracy'] for r in rs),'per_lang_median':{l:statistics.median(r['per_language_accuracy'][l] for r in rs) for l in rs[0]['per_language_accuracy']},'seeds':ss}
        paired=sorted(set(cvout['random']['seeds'])&set(cvout['joint']['seeds']));ds=[transfer['joint-s'+s]['accuracy']-transfer['random-s'+s]['accuracy'] for s in paired]
        cvout['paired_joint_minus_random']={'seeds':paired,'acc_diffs':ds,'median':statistics.median(ds),'n_pos':sum(d>0 for d in ds),
            'trimmed_diffs':[transfer['joint-s'+s]['trimmed_accuracy']-transfer['random-s'+s]['trimmed_accuracy'] for s in paired]}
        cvout['note']=ref['note'];compare(ref,cvout,'TRANSFER_PARTIAL',report)
        base=recomputed_summaries['worldcuisines']['base-noadapter']
        report['cached_evaluations']={'main_runs':163,'main_rows':586800,'transfer_runs':27,'transfer_rows':48600,'missing_planned_transfer_runs':46}
        report['table2']=table2;report['table3']=t3;report['cvqa']=cvout
        report['scoring_audit']={'full_generation_accuracy':base['accuracy'],'conformance_sample_size':len(base['conformance_ids']),
                                'disagreements':len(base['conformance_failures']),'shortcut_full_probability_replay':'UNAVAILABLE'}
        report['status']='PASS'
    except Exception as exc:
        report['status']='FAIL';report['error']=type(exc).__name__+': '+str(exc)
    report['seconds']=round(time.time()-start,3)
    (out/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','numeric_comparisons','seconds']}))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
