"""Audit completed paired studies before reporting efficiency or quality yields.

A running/failed study never receives a final speed ratio. Repetitions measure
run variability on fixed images; they are not independent dataset samples.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'egoworld/experiments'), str(REPO/'egoworld/src')]
from output_quality import evaluate


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid measured duration')
    return value


def distribution(values):
    values = list(values)
    if not values:
        raise ValueError('Empty measurement distribution')
    return {'n': len(values), 'median': statistics.median(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values), 'values': values}


def paired_ratios(rows, reference, compared, required):
    """Ratios only for every predeclared, successful exact-output pair."""
    groups = defaultdict(dict)
    for row in rows:
        if row['track'] != 'fixed_models' or row['role'] != 'development_repeats':
            continue
        if not row['full_parity'] or row['status'] != 'passed':
            raise ValueError('No exact-output ratio for incomplete or mismatched trials')
        if row['name'] in groups[row['group']]:
            raise ValueError('Duplicate repeat identity')
        groups[row['group']][row['name']] = positive(row['wall_seconds'])
    if len(groups) != required or any(reference not in g or compared not in g for g in groups.values()):
        raise ValueError('Not all predeclared pairs are available')
    ratios = [g[compared]/g[reference] for _, g in sorted(groups.items())]
    return {**distribution(ratios), 'definition': f'{compared} seconds / {reference} seconds',
        'scope': 'Within-host fixed-image repeated-run variability only; not independent data/model generalization',
        'reference_faster_pairs': sum(r > 1 for r in ratios)}


def audit_quality(folder, group, spec, summary):
    quality = read(folder/'quality.json')
    corpus = read(group['corpus'])
    if quality['corpus_sha256'] != digest(group['corpus']):
        raise ValueError('Changed quality corpus')
    if quality['policy'] != group['quality_policy']:
        raise ValueError('Changed quality policy')
    for job in spec['jobs']:
        job_folder = folder/job['name']
        for mode, prefix in [('exact_frames','exact'), ('identical_image_content','identical-image')]:
            score = quality[mode][job['name']]
            common = Path(score['common_output_file'])
            if digest(common) != score['common_output_sha256']:
                raise ValueError('Changed common product artifact')
            predictions_path = job_folder/(prefix+'-quality-predictions.json' if job['kind']=='datajuicer_native' else 'quality-predictions.json')
            predictions = read(predictions_path)['predictions']
            recomputed = evaluate(predictions, corpus['ground_truth'], corpus['samples'], group['quality_policy'])
            if any(score.get(k) != v for k,v in recomputed.items()):
                raise ValueError('Saved quality differs from recomputed GT evaluation')
            if any(score.get(k) != v for k,v in summary['quality_summary'][mode][job['name']].items()):
                raise ValueError('Quality differs from controller completion receipt')
            if len(read(common)['predictions']) != score['common_output_rows']:
                raise ValueError('Common product row coverage changed')
            if abs(score['common_output_ready_seconds']-(score['native_wall_seconds']+score['normalization_seconds'])) > 1e-8:
                raise ValueError('Invalid common-output timing arithmetic')
    return quality


def analyse(study):
    study=Path(study); state=read(study/'study-state.json'); plan=read(study/'plan.json')
    if state['status'] != 'completed' or state.get('active_group') is not None:
        raise ValueError('Study is not terminal-completed; no final efficiency report is eligible')
    if len(state['groups']) != len(plan['groups']):
        raise ValueError('Missing predeclared groups')
    summaries={s['name']:s for s in state['groups']}
    if len(summaries)!=len(plan['groups']):
        raise ValueError('Duplicate group receipts')
    rows=[]; source_identity=None; model_identity=None
    for group in plan['groups']:
        name=group['name']; summary=summaries[name]; folder=study/name; spec=read(group['spec']); report=read(folder/'report.json')
        if summary.get('audit_status') != 'passed' or digest(folder/'report.json')!=summary['raw_report_sha256']:
            raise ValueError('Incomplete or changed raw group receipt')
        if not all(report.get(k) for k in ['sources_unchanged','inputs_unchanged','configs_unchanged','dependencies_unchanged']):
            raise ValueError('Changed frozen study identity')
        if source_identity is None: source_identity=report['source_hashes']
        if report['source_hashes']!=source_identity:
            raise ValueError('Different measured core sources across groups')
        if [t['name'] for t in report['trials']] != [j['name'] for j in spec['jobs']]:
            raise ValueError('Missing/reordered measured jobs')
        exact=False
        if group['track']=='fixed_models':
            parity=read(folder/'full-parity.json')
            if parity!=summary['full_parity'] or not parity['passed'] or set(parity['comparisons'])!={j['name'] for j in spec['jobs'] if j['name']!=parity['reference']} or not all(p['passed'] for p in parity['comparisons'].values()):
                raise ValueError('Missing full-chain parity evidence')
            exact=True
            model_manifest=next((folder/group['reference_job']/'worker/runs/run_id=c1/model-output').glob('run_id=*/run_manifest.json'))
            current_models=read(model_manifest)['provenance']['local_assets']
            if model_identity is None:model_identity=current_models
            if current_models!=model_identity:raise ValueError('Different model assets across fixed-model repetitions')
        quality=audit_quality(folder,group,spec,summary)
        for trial in report['trials']:
            t={k:trial.get(k) for k in ['name','kind','status','wall_seconds','process_tree_peak_rss_bytes','gpu_whole_device_peak_used_bytes']}
            positive(t['wall_seconds'])
            t.update(group=name,track=group['track'],role=group['role'],source_images=group['input_source_image_count'],full_parity=exact,quality={})
            for mode in ['exact_frames','identical_image_content']:
                s=quality[mode][t['name']]; duration=positive(s['common_output_ready_seconds'])
                if s['native_wall_seconds'] != t['wall_seconds']:
                    raise ValueError('Mismatched raw/common duration')
                t['quality'][mode]={k:s[k] for k in ['gt_hands','matched_hands','missed_hands','false_positives','mpjpe_root_mm_detected_only','pa_mpjpe_mm_detected_only','reprojection_mean_px_detected_only','pck_all_gt','quality_hands','common_output_rows','normalization_seconds','common_output_ready_seconds']}
                t['quality'][mode]['quality_hands_per_second_common']={threshold:count/duration for threshold,count in s['quality_hands'].items()}
                t['quality'][mode]['quality_hands_per_second_raw']={threshold:count/t['wall_seconds'] for threshold,count in s['quality_hands'].items()}
            rows.append(t)
    expected_native=plan['requirements']['native_product_repeats']; expected_fixed=plan['requirements']['fixed_model_pairs_per_reference']
    buckets=defaultdict(list)
    for row in rows:
        if row['role']=='development_repeats': buckets[(row['track'],row['name'])].append(row)
    aggregate=[]
    for (track,name),bucket in sorted(buckets.items()):
        required=expected_fixed if track=='fixed_models' else expected_native
        if len(bucket)!=required: raise ValueError('Missing planned repetitions')
        aggregate.append({'track':track,'name':name,'n':len(bucket),'wall_seconds':distribution(r['wall_seconds'] for r in bucket),'common_output_ready_seconds':distribution(r['quality']['exact_frames']['common_output_ready_seconds'] for r in bucket),'quality_yield':{mode:{threshold:distribution(r['quality'][mode]['quality_hands_per_second_common'][threshold] for r in bucket) for threshold in ['20','50']} for mode in ['exact_frames','identical_image_content']}})
    ratios={name:paired_ratios(rows,'egoworld',name,expected_fixed) for name in ['shared-process','shared-inprocess']}
    boot_blocks=read(study/'boot-blocks.json') if (study/'boot-blocks.json').exists() else {}
    for row in rows:
        row['environment_block']=boot_blocks.get(row['group'],{}).get('block','uninterrupted_initial_study')
    if boot_blocks:
        for ratio in ratios.values():ratio['environment_note']='Five original predeclared paired groups span explicitly recorded pre/post external restart blocks; do not claim one uninterrupted host/cache/thermal state.'
    return {'status':'completed_audited','study':str(study.resolve()),'groups':len(plan['groups']),'trials':len(rows),'plan_sha256':state['plan_sha256'],'rows':rows,'environment_blocks':boot_blocks,'external_interruption':plan.get('external_interruption'),'fixed_model_assets':model_identity,'development_distributions':aggregate,'eligible_fixed_model_ratios':ratios,'limits':['Native product models/device/export work differ; native latency ratios are not framework speedups.','Exact source frames are primary; identical static-image content is a predeclared supplementary score.','Three/five repeats quantify this host and fixed corpus; they do not increase independent sample count or prove a population confidence interval.','1/4-image prefix points are single runs on nested image subsets; no isolated throughput scaling law.','EgoWorld FreiHAND side is a right-hand task prior; native predicts side. Do not generalize side accuracy.','MPJPE averages are conditional on matched detections; coverage, PCK and qualified counts keep all GT in the denominator.','Additional predictions are false positives for this single-target output task, not an independent annotation audit of every background hand.','All root coordinates use model-prior scale; no measured world-scale accuracy is established.','Physical 2–16 GPU results are unavailable; server placement benefits remain hypotheses.']}


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for axis,track,title in [(axes[0],'fixed_models','Same SAM2 / HaMeR / Dex, full output parity, n=5'),(axes[1],'native_product','Native hand products, different models, n=3')]:
        series=[x for x in result['development_distributions'] if x['track']==track]
        labels=[x['name'] for x in series]
        for offset,key,label,color in [(-.18,'wall_seconds','Whole-entry','#3976a8'),(.18,'common_output_ready_seconds','Common hand output ready','#b49a64')]:
            med=[x[key]['median'] for x in series]
            err=[[x[key]['median']-x[key]['min'] for x in series],[x[key]['max']-x[key]['median'] for x in series]]
            axis.barh([i+offset for i in range(len(series))],med,height=.33,xerr=err,label=label,color=color)
        axis.set_yticks(range(len(series)),labels);axis.invert_yaxis();axis.set_title(title,fontsize=10);axis.set_xlabel('seconds; median and observed min-max');axis.legend(fontsize=8)
    fig.suptitle('Separate execution and product comparisons on 12 development images')
    for extension in ['png','svg']:fig.savefig(Path(output)/('latency.'+extension),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    native=[r for r in result['rows'] if r['track']=='native_product' and r['role']=='development_repeats']
    names=sorted({r['name'] for r in native})
    for axis,mode,title in [(axes[0],'exact_frames','Exact source frames (primary)'),(axes[1],'identical_image_content','Identical static images (supplementary)')]:
        for name in names:
            rows=[r for r in native if r['name']==name]
            x=[r['quality'][mode]['common_output_ready_seconds'] for r in rows]
            y=[r['quality'][mode]['quality_hands']['20']/r['quality'][mode]['gt_hands'] for r in rows]
            axis.scatter(x,y,alpha=.75,s=25,label=name)
        axis.legend(loc='lower right',fontsize=8)
        axis.set_ylim(0,1.12);axis.set_xlabel('Common hand product ready (seconds)');axis.set_ylabel('20 mm qualified hands / all GT hands');axis.set_title(title,fontsize=10);axis.grid(alpha=.2)
    fig.suptitle('Native product quality and cost: repeats are the same 12 source images')
    for extension in ['png','svg']:fig.savefig(Path(output)/('quality-cost.'+extension),dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots", action="store_true", help="Also render figures (requires matplotlib)")
    args = parser.parse_args()
    result = analyse(args.study)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plots:
        plot(result, destination)
    print(json.dumps({"status": result["status"], "groups": result["groups"],
                      "trials": result["trials"], "output": str(destination)}))


if __name__ == "__main__":
    main()
