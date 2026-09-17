--- CODE CELL 2 ---
import os
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Config:
    variant: str = 'consensus'
    data_dir: str = ''
    # Temporary intermediate files are outside Kaggle's submitted output directory.
    output_dir: str = '/kaggle/temp/poker7_05_gated_consensus'
    submission_path: str = '/kaggle/working/submission.csv'
    seed: int = 20260909
    threads: int = min(4, os.cpu_count() or 2)
    batch_rows: int = 250_000
    unknown_per_pool: int = 160
    unknown_min_shared: int = 15
    n_folds: int = 5
    hand_trees: int = 420
    rank_trees: int = 420
    risk_trees: int = 620
    ap5_trees: int = 280
    generic_hand_trees: int = 300
    mechanism_trees: int = 260
    pair_blend: tuple = (.15, .70, .15)
    open_set_weight: float = 0.0
    pair_policy_search: bool = True
    temporal_tail_audit: bool = True
    pair_catboost: bool = True
    pair_catboost_trees: int = 420
    novelty_audit: bool = True
    pu_unknown_weight: float = 0.04
    bag_seeds: tuple = (20260909, 137)
    # No score cut-off, no planted-evidence insertion, no heuristic retrieval pruning.
    full_scan: bool = True
    temporal_augmentation: bool = True
    archive_diagnostics: bool = True
    other_enabled: bool = True
    llm_gguf_path: str = ''
    llm_cases: int = 5
    llm_max_tokens: int = 320
    fixture_mode: bool = False
    query_trees: int = 360
    map_trees: int = 480
    witness_trees: int = 320
    challenger_pair_trees: int = 480
    nested_pair_selection: bool = True
    v6_pair_weight: float = 0.0  # selected on inner nested holdout, never outer fold 0
    gate_min_queries: int = 12
    gate_min_pools: int = 6
    keep_anchor_csv: bool = True
    run_preflight: bool = True

CFG = Config()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = str(CFG.threads)
os.environ['CUDA_VISIBLE_DEVICES'] = ''



--- CODE CELL 4 ---
import gc, json, math, time, hashlib, importlib, platform, warnings, types
from types import SimpleNamespace
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import joblib
from numba import njit
from scipy.special import expit, softmax
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import lightgbm as lgb


PIPELINE_VERSION = 'sentinel-cpu-7.0.0-consensus'
SOURCE_SHA256 = 'POKER_SENTINEL_V7_CONSENSUS'
FAMILIES = np.array(['directed_transfer', 'soft_play', 'coordinated_isolation'])
ALLOWED_BEHAVIORS = {'none', *FAMILIES.tolist(), 'other_coordination'}
REQUIRED = ['actions.parquet', 'development_evidence.csv', 'development_labels.csv', 'evaluation_pairs.csv', 'hands.parquet', 'players.parquet', 'sample_submission.csv', 'seats.parquet']
CARD = {r+s: i*4+j for i,r in enumerate('23456789TJQKA') for j,s in enumerate('cdhs')}
CARD_INV = {v:k for k,v in CARD.items()}
ACTION = {'fold':0, 'check':1, 'call':2, 'bet':3, 'raise':4, 'all_in':5}
STREET = {'preflop':0, 'flop':1, 'turn':2, 'river':3}
STREET_INV = {v:k for k,v in STREET.items()}
ACTION_INV = {v:k for k,v in ACTION.items()}
RUN_LOG = []
RUN_START = time.perf_counter()
OUT = Path(CFG.output_dir)


def log(stage, **values):
    import resource
    entry = {'stage': stage, 'elapsed_seconds': round(time.perf_counter()-RUN_START, 3), 'process_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2), **values}
    RUN_LOG.append(entry)
    print(json.dumps(entry, default=str), flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v,'item') else str(v)))
    tmp.replace(path)

def zip_inventory(path):
    from zipfile import ZipFile,BadZipFile
    try:
        with ZipFile(path) as z:
            hits={n:[i for i in z.infolist() if not i.is_dir() and Path(i.filename).name==n] for n in REQUIRED}
            if all(len(v)==1 for v in hits.values()):return {k:v[0] for k,v in hits.items()}
    except (OSError,BadZipFile):pass
    return None


def unpack_dataset(path,cfg):
    """Extract only eight expected files, by basename, never arbitrary archive paths."""
    from zipfile import ZipFile
    import shutil
    inventory=zip_inventory(path)
    if inventory is None:raise ValueError('Archive must contain exactly one copy of every competition file')
    if sum(i.file_size for i in inventory.values())>6*1024**3:
        raise ValueError('Archive exceeds the 6 GiB declared extraction guard; inspect it explicitly')
    key=sha256_file(path)[:20];dest=Path(cfg.output_dir)/('dataset_'+key);dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'EXTRACT_COMPLETE.json'
    if marker.exists() and all((dest/n).is_file() and (dest/n).stat().st_size==inventory[n].file_size for n in REQUIRED):return dest
    with ZipFile(path) as z:
        for name,info in inventory.items():
            tmp=dest/(name+'.partial')
            with z.open(info) as src,open(tmp,'wb') as sink:shutil.copyfileobj(src,sink,length=2**20)
            if tmp.stat().st_size!=info.file_size:raise IOError('Truncated ZIP member: '+name)
            tmp.replace(dest/name)
    save_json(marker,{'archive_sha256':sha256_file(path),'extracted':REQUIRED})
    return dest


def discover_data(explicit=''):
    if explicit:
        p=Path(explicit)
        if p.is_file() and p.suffix.lower()=='.zip':return unpack_dataset(p,CFG)
        if not p.is_dir():raise FileNotFoundError(f'Dataset path does not exist: {p}')
        missing=[n for n in REQUIRED if not (p/n).is_file()]
        if missing:raise FileNotFoundError(f'Missing from {p}: {missing}')
        return p
    roots=[];archives=[]
    for base in (Path('/kaggle/input'),Path('/mnt/data')):
        if not base.exists():continue
        for p in base.rglob('evaluation_pairs.csv'):
            if all((p.parent/n).is_file() for n in REQUIRED):roots.append(p.parent)
        # Archive inspection reads directories only; non-dataset bundles are ignored.
        for p in base.rglob('*.zip'):
            if zip_inventory(p) is not None:archives.append(p)
    roots=list(dict.fromkeys(roots));archives=list(dict.fromkeys(archives))
    if len(roots)==1:return roots[0]
    if len(roots)==0 and len(archives)==1:return unpack_dataset(archives[0],CFG)
    raise RuntimeError(f'Ambiguous/missing dataset: folders={roots}, archives={archives}. Set CFG.data_dir to one folder or ZIP.')


def parquet_batches(path, columns, batch_rows):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError('pyarrow is required. Attach its wheel offline or install pyarrow in an Internet-enabled setup session.') from e
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        yield b.to_pandas()

def check_dependencies():
    versions = {}
    for package in ['numpy','pandas','numba','scipy','sklearn','lightgbm','joblib','pyarrow','xgboost']:
        try: versions[package] = getattr(importlib.import_module(package), '__version__', 'present')
        except ImportError:
            if package == 'pyarrow' and CFG.fixture_mode: versions[package] = 'not needed for in-memory fixture'
            else: raise ImportError(f'Missing dependency: {package}. No automatic package upgrades are performed.')
    try: versions['catboost'] = importlib.import_module('catboost').__version__
    except ImportError: versions['catboost'] = 'not installed; optional pair expert disabled'
    versions['python'] = platform.python_version()
    save_json(OUT/'environment.json', versions)
    log('environment', versions=versions, cpu_threads=CFG.threads, gpu=False)
    return versions



--- CODE CELL 6 ---
A_DTYPE = np.dtype([('hid','i4'),('ano','i4'),('player','i4'),('street','i1'),('kind','i1'),
                    ('amount','f4'),('amount_to','f4'),('pot','f4'),('stack','f4'),('call','f4'),('active','i1')])
S_COLUMNS = ['starting_stack','total_contribution','net_chips','won_share','folded','went_to_showdown']



def make_array(root, name, shape, dtype, fill=None):
    arr = np.lib.format.open_memmap(root/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
    if fill is not None: arr[:] = fill
    return arr

def runtime_source_fingerprint():
    """Hash executable code without cell filenames; edited notebook code invalidates caches."""
    def norm(value):
        if isinstance(value, types.CodeType):
            return {'bytecode':value.co_code.hex(),'constants':[norm(v) for v in value.co_consts],
                    'names':value.co_names,'variables':value.co_varnames,'freevars':value.co_freevars,
                    'argcount':value.co_argcount,'kwargcount':value.co_kwonlyargcount}
        if isinstance(value,(tuple,list)):return [norm(v) for v in value]
        if isinstance(value,(set,frozenset)):return sorted([norm(v) for v in value],key=repr)
        if isinstance(value,(str,int,float,bool)) or value is None:return value
        return repr(value)
    records={}
    for name,obj in sorted(list(globals().items())):
        fun=getattr(obj,'py_func',obj)
        if isinstance(fun,types.FunctionType) and fun.__module__==__name__:
            records[name]=[norm(fun.__code__),norm(fun.__defaults__)]
        elif isinstance(obj,type) and obj.__module__==__name__:
            records[name]={k:[norm(v.__code__),norm(v.__defaults__)] for k,v in sorted(obj.__dict__.items()) if isinstance(v,types.FunctionType)}
    value=[SOURCE_SHA256,records,globals().get('HAND_FEATURES'),globals().get('PAIR_FEATURES')]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def prepare_pack(data_dir, cfg, raw_frames=None):
    """raw_frames is ONLY for explicit, local mechanical tests."""
    fixture = raw_frames is not None
    if fixture and not cfg.fixture_mode: raise ValueError('Fixture input requires fixture_mode=True')
    if fixture:
        signature = 'fixture-' + str(cfg.seed)
        metadata = {'kind':'SYNTHETIC_MECHANICAL_FIXTURE'}
    else:
        metadata = {f: {'bytes':(data_dir/f).stat().st_size, 'sha256':sha256_file(data_dir/f)} for f in REQUIRED}
        signature = hashlib.sha256(json.dumps([metadata, runtime_source_fingerprint(), asdict(cfg)], sort_keys=True).encode()).hexdigest()[:20]
    root = Path(cfg.output_dir)/('cache_'+signature)
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'COMPLETE.json'
    if marker.exists() and not fixture:
        meta = joblib.load(root/'metadata.joblib')
        p = SimpleNamespace(**meta, root=root)
        for name in ['seat_players','seat_cards','seat_values','board','hand_values','times','actions','action_ptr','player_local','player_table','pair_counts']:
            setattr(p, name, np.load(root/f'{name}.npy', mmap_mode='r'))
        log('compact_cache_reused', path=str(root))
        return p
    def read(name):
        if fixture: return raw_frames[name].copy()
        return pd.read_csv(data_dir/name, dtype={'pair_id':str,'hand_id':str,'player_1':str,'player_2':str}) if name.endswith('.csv') else pd.read_parquet(data_dir/name)
    hands = read('hands.parquet')
    players = read('players.parquet')
    labels = read('development_labels.csv')
    evidence = read('development_evidence.csv')
    epairs = read('evaluation_pairs.csv')
    sample = read('sample_submission.csv')
    assert set(labels['label'].unique()).issubset({0,1}), 'Public labels must be 0/1'
    assert labels['pair_id'].is_unique and epairs['pair_id'].is_unique and sample['pair_id'].is_unique
    assert set(sample['pair_id']) == set(epairs['pair_id'])
    assert set(labels.loc[labels.label.eq(1),'behavior_family']).issubset(set(FAMILIES))
    assert set(hands.phase.unique()) == {'development','evaluation'}
    hi = pd.Index(hands.hand_id.astype(str)); pi = pd.Index(players.player_id.astype(str))
    assert hi.is_unique and pi.is_unique
    h = len(hands); nplayers = len(players)
    tables = pd.Index(pd.unique(hands.table_id.astype(str)))
    tablecodes = tables.get_indexer(hands.table_id)
    phasecodes = hands.phase.map({'development':0,'evaluation':1}).to_numpy(np.int8)
    assert hands.players_dealt.eq(6).all()
    times = make_array(root,'times',(h,),np.int64)
    times[:] = pd.to_datetime(hands.started_at, utc=True).astype('int64').to_numpy()
    hv = make_array(root,'hand_values',(h,7),np.float64)
    hv[:] = np.column_stack([hands.big_blind, hands.small_blind, hands.button_seat, hands.final_pot,
                            hands.players_at_showdown, phasecodes, tablecodes])
    board = make_array(root,'board',(h,5),np.int8,-1)
    for i, b in enumerate(hands.board_cards.fillna('').astype(str)):
        cc = b.split()
        if len(cc) not in (0,3,4,5): raise ValueError(f'Unexpected board length at hand row {i}')
        for j,c in enumerate(cc): board[i,j] = CARD[c]
    sp = make_array(root,'seat_players',(h,6),np.int32,-1)
    sc = make_array(root,'seat_cards',(h,6,2),np.int8,-1)
    sv = make_array(root,'seat_values',(h,6,6),np.float32,0)
    seat_seen = np.zeros((h,6), dtype=np.uint8)
    cols = ['hand_id','player_id','seat_no','hole_card_1','hole_card_2',*S_COLUMNS]
    batches = [raw_frames['seats.parquet']] if fixture else parquet_batches(data_dir/'seats.parquet',cols,cfg.batch_rows)
    for b in batches:
        hh=hi.get_indexer(b.hand_id); pp=pi.get_indexer(b.player_id); ss=b.seat_no.to_numpy(np.intp)
        assert (hh>=0).all() and (pp>=0).all() and ((ss>=0)&(ss<6)).all()
        flat=hh*6+ss
        if len(np.unique(flat)) != len(flat) or seat_seen[hh,ss].any(): raise ValueError('Duplicate seat key')
        seat_seen[hh,ss]=1
        sp[hh,ss]=pp
        sc[hh,ss,0]=b.hole_card_1.map(CARD).to_numpy(np.int8)
        sc[hh,ss,1]=b.hole_card_2.map(CARD).to_numpy(np.int8)
        sv[hh,ss]=b[S_COLUMNS].to_numpy(np.float32)
    assert seat_seen.all() and (sp>=0).all(), 'Missing seat rows'
    del seat_seen
    assert np.max(np.abs(sv[:,:,2].sum(axis=1))) < 1e-3, 'Chip conservation failed'
    assert np.allclose(sv[:,:,1].sum(axis=1),hv[:,3]), 'Pot/contribution mismatch'
    player_table = make_array(root,'player_table',(nplayers,),np.int32,-1)
    # Membership must be derived from gameplay, never inferred from account IDs.
    for t in range(len(tables)):
        pp = np.unique(sp[tablecodes==t].ravel())
        if (player_table[pp]>=0).any(): raise ValueError('A player occurs in multiple persistent pools')
        player_table[pp]=t
    assert (player_table>=0).all()
    max_pool = int(max(np.sum(player_table==t) for t in range(len(tables))))
    player_local = make_array(root,'player_local',(nplayers,),np.int32,-1)
    for t in range(len(tables)):
        pp=np.flatnonzero(player_table==t); player_local[pp]=np.arange(len(pp))
    acols=['hand_id','action_no','player_id','street','action','amount','amount_to','pot_before','stack_before','to_call','players_active']
    if fixture: na=len(raw_frames['actions.parquet'])
    else:
        import pyarrow.parquet as pq
        na=pq.ParquetFile(data_dir/'actions.parquet').metadata.num_rows
    au=make_array(root,'actions_unsorted',(na,),A_DTYPE)
    batches=[raw_frames['actions.parquet']] if fixture else parquet_batches(data_dir/'actions.parquet',acols,cfg.batch_rows)
    offset=0
    for b in batches:
        n=len(b); sl=slice(offset,offset+n); hh=hi.get_indexer(b.hand_id); pp=pi.get_indexer(b.player_id)
        assert (hh>=0).all() and (pp>=0).all()
        au['hid'][sl]=hh; au['player'][sl]=pp; au['ano'][sl]=b.action_no
        for target,source,mapping in [('street','street',STREET),('kind','action',ACTION)]:
            z=b[source].map(mapping)
            if z.isna().any(): raise ValueError(f'Unknown {source}')
            au[target][sl]=z
        for target,source in [('amount','amount'),('amount_to','amount_to'),('pot','pot_before'),('stack','stack_before'),('call','to_call'),('active','players_active')]:
            au[target][sl]=b[source]
        offset+=n
    assert offset==na
    order=np.lexsort((au['ano'],au['hid']))
    aa=make_array(root,'actions',(na,),A_DTYPE)
    for start in range(0,na,cfg.batch_rows): aa[start:start+cfg.batch_rows]=au[order[start:start+cfg.batch_rows]]
    aa.flush(); del order,au
    (root/'actions_unsorted.npy').unlink(missing_ok=True)
    if na>1:
        assert not np.any((aa['hid'][1:]==aa['hid'][:-1])&(aa['ano'][1:]==aa['ano'][:-1])), 'Duplicate action key'
    ptr=make_array(root,'action_ptr',(h+1,),np.int64)
    ptr[0]=0; ptr[1:]=np.cumsum(np.bincount(aa['hid'],minlength=h))
    counts = make_array(root,'pair_counts',(len(tables),2,max_pool,max_pool),np.int32,0)
    count_pair_exposures(sp, hv, player_local, counts)
    meta=dict(hand_ids=hi.to_numpy(),player_ids=pi.to_numpy(),table_ids=tables.to_numpy(),
              labels=labels,evidence=evidence,eval_pairs=epairs,sample=sample,max_pool=max_pool,fixture=fixture)
    joblib.dump(meta,root/'metadata.joblib',compress=3)
    save_json(root/'input_manifest.json',metadata)
    for a in [sp,sc,sv,board,hv,times,aa,ptr,player_table,player_local,counts]: a.flush()
    save_json(marker,{'version':PIPELINE_VERSION,'rows_hands':h,'rows_actions':na,'fixture':fixture})
    p=SimpleNamespace(**meta,root=root,seat_players=sp,seat_cards=sc,seat_values=sv,board=board,
                      hand_values=hv,times=times,actions=aa,action_ptr=ptr,
                      player_table=player_table,player_local=player_local,pair_counts=counts)
    log('compact_data_ready', hands=h,actions=na,players=nplayers,pools=len(tables),fixture=fixture)
    return p

@njit(cache=False)
def count_pair_exposures(seat_players, hv, player_local, counts):
    for h in range(len(seat_players)):
        t=int(hv[h,6]); phase=int(hv[h,5])
        for a in range(6):
            u=player_local[seat_players[h,a]]
            for b in range(a+1,6):
                v=player_local[seat_players[h,b]]
                counts[t,phase,u,v]+=1; counts[t,phase,v,u]+=1



--- CODE CELL 8 ---
@njit(cache=False)
def straight_high(mask):
    for hi in range(14,5-1,-1):
        if hi==5:
            if (mask & 4111)==4111: return 5
        else:
            pattern=31 << (hi-6)
            if (mask&pattern)==pattern: return hi
    return 0

@njit(cache=False)
def rank_key(category, r0=0,r1=0,r2=0,r3=0,r4=0):
    return category*759375+r0*50625+r1*3375+r2*225+r3*15+r4

@njit(cache=False)
def made_rank(cards, n):
    counts=np.zeros(15,np.int8); suits=np.zeros(4,np.int8); masks=np.zeros(4,np.int32); mask=0
    for i in range(n):
        c=int(cards[i]); r=c//4+2; s=c%4
        counts[r]+=1; suits[s]+=1; masks[s]|=1<<(r-2); mask|=1<<(r-2)
    flush=-1
    for s in range(4):
        if suits[s]>=5:
            flush=s; st=straight_high(masks[s])
            if st: return rank_key(8,st)
    qu=0; tr=0; pair=0
    for r in range(14,1,-1):
        if counts[r]==4: qu=r
        if counts[r]>=3 and tr==0: tr=r
    if qu:
        k=0
        for r in range(14,1,-1):
            if r!=qu and counts[r]>0: k=r; break
        return rank_key(7,qu,k)
    if tr:
        for r in range(14,1,-1):
            if r!=tr and counts[r]>=2: pair=r; break
        if pair: return rank_key(6,tr,pair)
    rr=np.zeros(5,np.int32)
    if flush>=0:
        k=0
        for r in range(14,1,-1):
            if masks[flush]&(1<<(r-2)):
                rr[k]=r; k+=1
                if k==5: break
        return rank_key(5,rr[0],rr[1],rr[2],rr[3],rr[4])
    st=straight_high(mask)
    if st: return rank_key(4,st)
    if tr:
        k=0
        for r in range(14,1,-1):
            if r!=tr and counts[r]>0:
                rr[k]=r; k+=1
                if k==2: break
        return rank_key(3,tr,rr[0],rr[1])
    pairs=np.zeros(3,np.int32); npair=0
    for r in range(14,1,-1):
        if counts[r]>=2:
            pairs[npair]=r; npair+=1
            if npair==3: break
    if npair>=2:
        k=0
        for r in range(14,1,-1):
            if r!=pairs[0] and r!=pairs[1] and counts[r]>0: k=r; break
        return rank_key(2,pairs[0],pairs[1],k)
    if npair==1:
        k=0
        for r in range(14,1,-1):
            if r!=pairs[0] and counts[r]>0:
                rr[k]=r;k+=1
                if k==3: break
        return rank_key(1,pairs[0],rr[0],rr[1],rr[2])
    k=0
    for r in range(14,1,-1):
        if counts[r]>0:
            rr[k]=r;k+=1
            if k==5: break
    return rank_key(0,rr[0],rr[1],rr[2],rr[3],rr[4])

@njit(cache=False)
def hole_proxy(c1,c2):
    r1=c1//4+2; r2=c2//4+2; hi=max(r1,r2); lo=min(r1,r2)
    v=0.10+0.030*hi+0.013*lo+0.26*(r1==r2)+0.05*(c1%4==c2%4)+0.035*(hi-lo<=2)
    return min(1.0,v)

@njit(cache=False)
def hand_context(c1,c2,board,nb):
    cards=np.zeros(7,np.int32); cards[0]=c1;cards[1]=c2
    for k in range(nb):cards[k+2]=board[k]
    if nb==0:
        strength=hole_proxy(c1,c2)
        return strength, int(strength*100000), 0, strength>=0.76, False
    value=made_rank(cards,nb+2); cat=value//759375
    br=made_rank(board,nb); bc=br//759375
    highboard=0
    for k in range(nb):highboard=max(highboard,int(board[k])//4+2)
    r1=c1//4+2;r2=c2//4+2
    top_pair=(r1==highboard or r2==highboard) and cat==1 and bc==0
    overpair=r1==r2 and r1>highboard
    strong=(cat>=2 and (cat>bc or value>br+50000)) or overpair or (top_pair and max(r1,r2)>=11)
    strength=min(1.0,0.15+0.115*cat+0.14*top_pair+0.12*overpair)
    suits=np.zeros(4,np.int8);mask=0
    for i in range(nb+2):suits[cards[i]%4]+=1;mask|=1<<(cards[i]//4)
    draw=False
    if nb<5:
        for s in range(4):
            if suits[s]==4 and cat<5:draw=True
        for hi in range(6,15):
            pat=31<<(hi-6); z=mask&pat; nbit=0
            while z: nbit+=z&1;z>>=1
            if nbit==4 and cat<4:draw=True
        z=mask&4111;nbit=0
        while z:nbit+=z&1;z>>=1
        if nbit==4 and cat<4:draw=True
    return strength,value,cat,strong,draw



--- CODE CELL 10 ---
HAND_FEATURES = '''pot_bb pair_contrib_bb contrib_gap_bb pair_pot_share pair_net_bb transfer_proxy_bb net_gap_bb
both_showdown one_fold both_fold min_stack_bb max_stack_bb hole_hi hole_lo hole_gap n_actions
aggressive_actions call_actions check_actions fold_actions raise_actions allin_call_actions allin_aggressive_actions
max_amount_bb max_call_bb max_amount_pot max_call_odds partner_facing partner_folds partner_calls partner_raises
partner_call_bb partner_strong_fold partner_weaker_call_bb hu_actions hu_checks hu_calls hu_aggression
hu_strong_passive hu_passive_both outsider_folds_livepair outsider_calls_livepair outsider_raises_livepair
duo_pressure member_aggr_to_outsiders strong_fold_to_outsider_partner_better weak_aggr_partner_strong
river_losing_call_bb river_winning_fold_bb postflop_strong_check surrender_contrib_bb max_made_category
min_made_category hu_mean_rank_gap member_strong_fold_any pressure_after_partner_fold postflop_actions
partner_live_actions partner_facing_pot_odds draw_passive_count aggr_member_min aggr_member_gap call_member_min
aggr_outside_imbalance weak_call_any_bb strong_agg_any fold_when_partner_strong max_stack_commit
partner_facing_big_raise hu_opportunity_stacks_live rule_directed rule_soft rule_isolation rule_other rule_any
flow_signed progress'''.split()
HAND_FEATURES += ['preflop_partner_faces', 'preflop_partner_folds', 'preflop_partner_calls', 'preflop_partner_raises', 'preflop_hu_checks_eligible', 'preflop_hu_calls_eligible', 'preflop_hu_bets_eligible', 'preflop_partner_ahead_fold', 'preflop_partner_behind_call_bb', 'preflop_outsider_pressure', 'flop_partner_faces', 'flop_partner_folds', 'flop_partner_calls', 'flop_partner_raises', 'flop_hu_checks_eligible', 'flop_hu_calls_eligible', 'flop_hu_bets_eligible', 'flop_partner_ahead_fold', 'flop_partner_behind_call_bb', 'flop_outsider_pressure', 'turn_partner_faces', 'turn_partner_folds', 'turn_partner_calls', 'turn_partner_raises', 'turn_hu_checks_eligible', 'turn_hu_calls_eligible', 'turn_hu_bets_eligible', 'turn_partner_ahead_fold', 'turn_partner_behind_call_bb', 'turn_outsider_pressure', 'river_partner_faces', 'river_partner_folds', 'river_partner_calls', 'river_partner_raises', 'river_hu_checks_eligible', 'river_hu_calls_eligible', 'river_hu_bets_eligible', 'river_partner_ahead_fold', 'river_partner_behind_call_bb', 'river_outsider_pressure', 'partner_raise_fraction_pot', 'partner_call_fraction_pot', 'partner_fold_fraction_pot', 'partner_call_commit_max', 'partner_fold_committed_max', 'partner_bet_excess_bb', 'partner_ahead_margin_sum', 'hu_dry_strong_checks', 'hu_drawing_checks', 'hu_tie_checks', 'hu_both_street_checks', 'outsider_double_pressure_streets', 'outsider_folds_after_duo', 'min_member_postflop_bets', 'isolation_partner_weak_call', 'isolation_partner_strong_fold', 'outsider_last_aggr_facing', 'outsider_facing_strong_fold', 'outsider_facing_weak_call_bb', 'outsider_facing_raise', 'outsider_facing_fold_odds_sum', 'pair_blind_count', 'pair_position_gap', 'board_paired_last_seen', 'board_monotone_last_seen', 'river_no_draw_losing_call', 'partner_still_live_fold_stack_ratio', 'partner_made_behind_shove_bb', 'partner_winning_surrender_bb', 'hu_street_count', 'hu_streets_no_aggression', 'pair_external_profit_per_contribution', 'hu_allin_passive_excluded', 'partner_both_stacks_live_faces']

# All 15 co-seated pairs are replayed for label-free within-hand attribution.
EXTRA_HAND = ['partner_raise_then_fold', 'partner_small_raise_fold_committed_bb', 'partner_check_after_raise', 'partner_river_ahead_fold_pot_bb', 'partner_river_losing_raise_bb', 'partner_low_odds_ahead_fold', 'hu_river_best_check', 'hu_river_worse_aggression', 'hu_strong_check_nonallin', 'partner_weak_call_large_fraction', 'outsider_best_made_fold_partner_better', 'outsider_worse_made_raise_partner_better', 'both_live_member_made_best_fold', 'both_live_member_made_worst_raise', 'partner_made_tie_call_bb', 'partner_made_tie_fold', 'partner_paid_after_prior_raise_bb', 'partner_made_ahead_call_bb', 'partner_made_ahead_raise_bb', 'partner_made_ahead_check', 'partner_fold_effective_stack_fraction', 'hu_river_made_tie_check', 'hu_river_board_plays_check', 'pair_made_best_versus_outsiders', 'member_raises_after_partner_call', 'outsider_fold_after_partner_flat', 'voluntary_vs_partner_action_count', 'voluntary_vs_outsider_action_count', 'unilateral_outsider_pressure_streets', 'reciprocal_outsider_pressure_streets', 'transfer_action_signed', 'surrender_action_signed', 'pressure_action_signed']
CONTRAST_SOURCES = ['rule_directed', 'rule_soft', 'rule_isolation', 'rule_other', 'partner_river_ahead_fold_pot_bb', 'partner_raise_then_fold', 'outsider_folds_after_duo', 'hu_strong_passive']
CONTRAST_FEATURES = [f'{s}_peer_{stat}' for s in CONTRAST_SOURCES for stat in ['rank','share','log_excess','top_margin']]
# ---- V4 action-reference extension (own information only in null predictions) ----
NULL_STATS = ['opportunities','surprisal_sum','surprisal_max','rare_actions',
              'fold_residual','check_residual','call_residual','aggression_residual',
              'sizing_z_max','sizing_z_sum','partner_card_aggression','partner_card_fold',
              'benefit_surprise','strong_fold_surprise','weak_aggression_surprise',
              'outside_support','reference_probability_min']
NULL_FEATURES = [f'null_{role}_{s}' for role in ('partner','outsider') for s in NULL_STATS]
NULL_FEATURES += ['null_benefit_max','null_surprise_max','null_dependency_abs',
                 'null_direct_support','null_external_support','null_action_score',
                 'null_rare_benefit_count','null_forced_passive_excluded']
NULL_CONTEXTS = 4*8*5*2
N_NULL_FEATURES = len(NULL_FEATURES)
NULL_FIELDS = 13  # n, four event counts, four size first moments, four second moments


NULL_START=len(HAND_FEATURES)+len(EXTRA_HAND)+len(CONTRAST_FEATURES)
HAND_FEATURES += EXTRA_HAND + CONTRAST_FEATURES + NULL_FEATURES
HF = {c:i for i,c in enumerate(HAND_FEATURES)}
MODEL_HAND = [c for c in HAND_FEATURES if c not in {'flow_signed','progress','transfer_action_signed','surrender_action_signed','pressure_action_signed'}]
MODEL_HAND_IDX = np.array([HF[c] for c in MODEL_HAND],np.int32)
RULES=['rule_directed','rule_soft','rule_isolation','rule_other']
RULE_IDX=np.array([HF[c] for c in RULES],np.int32)
assert len(set(HAND_FEATURES))==len(HAND_FEATURES) and HF['rule_directed']==70 and HF['flow_signed']==75
CONTRAST_IDX = np.array([HF[s] for s in CONTRAST_SOURCES], np.int32)
EXTRA_START=151
CONTRAST_START=151+len(EXTRA_HAND)
N_HAND_FEATURES=len(HAND_FEATURES)



@njit(cache=False)
def replay_block(hids, sp, sc, sv, board, hv, times, aa, ptr, plocal, pairmap, pair_lows, pair_highs):
    nout=len(hids)*15; nf=N_HAND_FEATURES; npairs=len(pair_lows); m=pairmap.shape[0]
    out=np.zeros((nout,nf),np.float32); out_p=np.zeros(nout,np.int32);out_h=np.zeros(nout,np.int32)
    total_prof=np.zeros((m,32,5),np.float64)
    shared_prof=np.zeros((npairs,2,32,5),np.float64)
    cursor=0
    if len(hids)==0:return out[:0],out_p[:0],out_h[:0],total_prof,shared_prof
    t0=times[hids[0]];t1=times[hids[-1]]
    for hh in range(len(hids)):
        h=hids[hh];bb=max(1.,hv[h,0]);button=int(hv[h,2]);pot=hv[h,3]
        live=np.ones(6,np.bool_);rem=sv[h,:,0].copy();comm=np.zeros(6,np.float64)
        sbseat=(button+1)%6;bbseat=(button+2)%6
        comm[sbseat]=min(rem[sbseat],hv[h,1]);comm[bbseat]=min(rem[bbseat],bb)
        rem-=comm
        member_agg=np.zeros(6,np.float64);member_calls=np.zeros(6,np.float64)
        member_out=np.zeros((15,2),np.float64);passive=np.zeros((15,2),np.float64)
        seat_a=np.zeros(15,np.int32);seat_b=np.zeros(15,np.int32);pidx=np.zeros(15,np.int32)
        x=np.zeros((15,nf),np.float64);nk=0
        for a in range(6):
            for b in range(a+1,6):
                u=plocal[sp[h,a]];v=plocal[sp[h,b]];p=pairmap[u,v]
                # Replay even unrequested pairs: the contrast reference must not depend on pair labels.
                # Orientation is storage only. Model features are symmetric.
                if sp[h,a] < sp[h,b]:sa=a;sb=b
                else:sa=b;sb=a
                seat_a[nk]=sa;seat_b[nk]=sb;pidx[nk]=p
                xa=sv[h,sa];xb=sv[h,sb]
                n1=xa[2]/bb;n2=xb[2]/bb;c1=xa[1]/bb;c2=xb[1]/bb
                f12=min(max(-n1,0.),max(n2,0.));f21=min(max(-n2,0.),max(n1,0.))
                ph1=hole_proxy(sc[h,sa,0],sc[h,sa,1]);ph2=hole_proxy(sc[h,sb,0],sc[h,sb,1])
                x[nk,0]=pot/bb;x[nk,1]=c1+c2;x[nk,2]=abs(c1-c2);x[nk,3]=(c1+c2)/(pot/bb+1e-6)
                x[nk,4]=n1+n2;x[nk,5]=max(f12,f21);x[nk,6]=abs(n1-n2)
                x[nk,7]=xa[5]*xb[5];x[nk,8]=xa[4]!=xb[4];x[nk,9]=xa[4]*xb[4]
                x[nk,10]=min(xa[0],xb[0])/bb;x[nk,11]=max(xa[0],xb[0])/bb
                x[nk,12]=max(ph1,ph2);x[nk,13]=min(ph1,ph2);x[nk,14]=abs(ph1-ph2)
                x[nk,75]=f12-f21;x[nk,76]=(times[h]-t0)/max(1,t1-t0)
                x[nk,138]=(sa==sbseat or sa==bbseat)+(sb==sbseat or sb==bbseat)
                x[nk,139]=min((sa-sb)%6,(sb-sa)%6)
                x[nk,148]=(n1+n2)/(c1+c2+1)
                nk+=1
        strength=np.zeros(6,np.float64);ranks=np.zeros(6,np.int64);cat=np.zeros(6,np.int32)
        strong=np.zeros(6,np.bool_);draw=np.zeros(6,np.bool_)
        prev_st=-1;last=-1
        street_checks=np.zeros((15,4,2),np.int32)
        street_bets=np.zeros((15,4,2),np.int32)
        hu_seen=np.zeros((15,4),np.int32)
        post_bets=np.zeros((15,2),np.float64)
        paired=False;monotone=False
        own_street_raises=np.zeros((6,4),np.int32)
        own_street_calls=np.zeros((6,4),np.int32)
        pair_out_street=np.zeros((15,4,2),np.int32)
        for z in range(ptr[h],ptr[h+1]):
            ac=aa[z];a=-1
            for s in range(6):
                if sp[h,s]==ac['player']:a=s;break
            if a<0:raise ValueError('Action actor is not seated in the hand')
            if not live[a]:raise ValueError('Folded actor takes another action')
            st=int(ac['street']);kind=int(ac['kind']);amt=float(ac['amount']);call=float(ac['call'])
            pb=max(float(ac['pot']),bb);stack=max(float(ac['stack']),0.)
            if st!=prev_st:
                if st<prev_st:raise ValueError('Non-chronological streets')
                nb=0 if st==0 else st+2
                for j in range(nb):
                    if board[h,j]<0:raise ValueError('Action uses an unrevealed/missing street')
                for s in range(6):
                    strength[s],ranks[s],cat[s],strong[s],draw[s]=hand_context(sc[h,s,0],sc[h,s,1],board[h],nb)
                last=-1;prev_st=st
                boardcnt=np.zeros(13,np.int32);suitcnt=np.zeros(4,np.int32)
                for bj in range(nb):
                    boardcnt[board[h,bj]//4]+=1;suitcnt[board[h,bj]%4]+=1
                paired=boardcnt.max()>=2
                monotone=suitcnt.max()>=3
            ag=(kind==3 or kind==4 or (kind==5 and amt>call+1e-6))
            ca=(kind==2 or (kind==5 and not ag))
            fo=(kind==0);ch=(kind==1);pas=ca or ch
            ctx=min(3,int(strength[a]*4))*8+st*2+int(call>0)
            loc=plocal[sp[h,a]]
            total_prof[loc,ctx,0]+=1;total_prof[loc,ctx,1]+=ag
            total_prof[loc,ctx,2]+=fo;total_prof[loc,ctx,3]+=ca;total_prof[loc,ctx,4]+=ch
            member_agg[a]+=ag;member_calls[a]+=ca
            odds=min(call,stack)/(pb+min(call,stack)+1e-6)
            for k in range(nk):
                s1=seat_a[k];s2=seat_b[k];p=pidx[k]
                member=(a==s1 or a==s2);both_live=live[s1] and live[s2]
                if not member:
                    if call>0 and (last==s1 or last==s2):
                        if both_live:
                            x[k,40]+=fo;x[k,41]+=ca;x[k,42]+=ag
                            if fo and ((last==s1 and own_street_calls[s2,st]>0) or (last==s2 and own_street_calls[s1,st]>0)):
                                x[k,EXTRA_START+25]+=1
                        elif fo:x[k,55]+=1
                        if both_live and fo and street_bets[k,st,0]>0 and street_bets[k,st,1]>0:
                            x[k,129]+=1
                    continue
                partner=s2 if a==s1 else s1;side=0 if a==s1 else 1
                if p>=0:
                    shared_prof[p,side,ctx,0]+=1;shared_prof[p,side,ctx,1]+=ag
                    shared_prof[p,side,ctx,2]+=fo;shared_prof[p,side,ctx,3]+=ca;shared_prof[p,side,ctx,4]+=ch

                e=EXTRA_START;orient=1. if side==0 else -1.
                active_outsiders=0;best_out=0
                for oside in range(6):
                    if oside!=s1 and oside!=s2 and live[oside]:
                        active_outsiders+=1;best_out=max(best_out,ranks[oside])
                if both_live:
                    facing_partner=call>0 and last==partner
                    stacks_can_bet=stack>0 and rem[partner]>0
                    # These comparisons use made hands on the current street, NOT future equity.
                    if facing_partner:
                        if fo and own_street_raises[a,st]>0:
                            x[k,e+0]+=1
                            if call/pb<.25:x[k,e+1]+=comm[a]/bb
                        x[k,e+9]+=(ca and ranks[a]<ranks[partner] and amt/pb>.40)
                        if st>0:
                            x[k,e+5]+=(fo and ranks[a]>ranks[partner] and odds<.15)
                            x[k,e+14]+=(ca and ranks[a]==ranks[partner])*amt/bb
                            x[k,e+15]+=(fo and ranks[a]==ranks[partner])
                            x[k,e+17]+=(ca and ranks[a]>ranks[partner])*amt/bb
                            x[k,e+18]+=(ag and ranks[a]>ranks[partner])*amt/bb
                        if st==3:
                            x[k,e+3]+=(fo and ranks[a]>ranks[partner])*pb/bb
                            x[k,e+4]+=(ag and ranks[a]<ranks[partner])*amt/bb
                        if ca and own_street_raises[a,st]>0:x[k,e+16]+=amt/bb
                        if fo:x[k,e+20]=max(x[k,e+20],min(call,stack)/max(bb,min(stack,rem[partner])))
                        if ca:x[k,e+30]+=orient*amt/bb*(.25+.75*(st>0 and ranks[a]<ranks[partner]))
                        if fo and (strong[a] or (st>0 and ranks[a]>ranks[partner])):x[k,e+31]+=orient*comm[a]/bb
                    if stacks_can_bet:
                        if last==partner or active_outsiders==0:x[k,e+26]+=1
                        elif last>=0:x[k,e+27]+=1
                        if ch and own_street_raises[a,st]>0:x[k,e+2]+=1
                        if ch and st>0 and ranks[a]>ranks[partner]:x[k,e+19]+=1
                    if active_outsiders==0 and stacks_can_bet:
                        x[k,e+8]+=(ch and strong[a] and st>0)
                        if st==3:
                            x[k,e+6]+=(ch and ranks[a]>ranks[partner])
                            x[k,e+7]+=(ag and ranks[a]<ranks[partner])
                            x[k,e+21]+=(ch and ranks[a]==ranks[partner])
                            x[k,e+22]+=(ch and ranks[a]==made_rank(board[h],5))
                    if active_outsiders>0:
                        if st>0:
                            x[k,e+10]+=(fo and ranks[a]>best_out and ranks[partner]>ranks[a])
                            x[k,e+11]+=(ag and ranks[a]<best_out and ranks[partner]>best_out)
                            x[k,e+12]+=(fo and ranks[a]>best_out and ranks[a]>ranks[partner])
                            x[k,e+13]+=(ag and ranks[a]<best_out and ranks[a]<ranks[partner])
                            x[k,e+23]+=(max(ranks[a],ranks[partner])>best_out)
                        if ag and last!=partner:
                            pair_out_street[k,st,side]+=1
                            x[k,e+32]+=orient*max(amt-call,0)/bb
                        if ag and own_street_calls[partner,st]>0:x[k,e+24]+=1
                x[k,15]+=1;x[k,16]+=ag;x[k,17]+=ca;x[k,18]+=ch;x[k,19]+=fo;x[k,20]+=(kind==4)
                x[k,21]+=(kind==5 and ca);x[k,22]+=(kind==5 and ag)
                x[k,23]=max(x[k,23],amt/bb);x[k,24]=max(x[k,24],call/bb)
                x[k,25]=max(x[k,25],min(20.,amt/pb));x[k,26]=max(x[k,26],odds)
                x[k,49]+=(st>0 and strong[a] and ch)
                x[k,51]=max(x[k,51],cat[a]);x[k,56]+=(st>0);x[k,57]+=live[partner]
                x[k,54]+=(fo and strong[a]);x[k,59]+=(draw[a] and pas)
                x[k,64]+=(ca and strength[a]<0.4)*amt/bb;x[k,65]+=(strong[a] and ag)
                x[k,67]=max(x[k,67],amt/max(stack,bb))
                if fo:x[k,50]+=comm[a]/bb
                if both_live and call>0 and last==partner:
                    x[k,27]+=1;x[k,28]+=fo;x[k,29]+=ca;x[k,30]+=ag
                    x[k,31]+=ca*amt/bb;x[k,32]+=(fo and strong[a])
                    x[k,33]+=(ca and ranks[a]<ranks[partner])*amt/bb
                    x[k,58]+=odds;x[k,68]=max(x[k,68],call/bb)
                    if st==3:
                        x[k,47]+=(ca and ranks[a]<ranks[partner])*amt/bb
                        x[k,48]+=(fo and ranks[a]>ranks[partner])*comm[a]/bb
                slot=77+10*st
                pf=both_live and call>0 and last==partner
                stklive=stack>0 and rem[partner]>0
                if pf:
                    x[k,slot]+=1;x[k,slot+1]+=fo;x[k,slot+2]+=ca;x[k,slot+3]+=ag
                    x[k,slot+7]+=(fo and ranks[a]>ranks[partner])
                    x[k,slot+8]+=(ca and ranks[a]<ranks[partner])*amt/bb
                    x[k,150]+=stklive
                    if ag:
                        x[k,117]=max(x[k,117],(amt-call)/pb)
                        x[k,122]+=(amt-call)/bb
                    if ca:
                        x[k,118]=max(x[k,118],amt/pb)
                        x[k,120]=max(x[k,120],amt/max(1.,stack))
                    if fo:
                        x[k,119]=max(x[k,119],call/pb)
                        x[k,121]=max(x[k,121],comm[a]/max(1.,comm[a]+stack))
                        x[k,143]=max(x[k,143],comm[a]/max(1.,stack))
                    if st>0:
                        x[k,123]+=(ranks[a]>ranks[partner])*(fo-ca)
                        x[k,144]+=(ag and ranks[a]<ranks[partner] and amt>=.90*stack)*amt/bb
                    if st==3:
                        x[k,142]+=(ca and ranks[a]<ranks[partner])
                        x[k,145]+=(fo and ranks[a]>ranks[partner])*comm[a]/bb
                if both_live and ag and last!=partner:
                    x[k,slot+9]+=1
                    street_bets[k,st,side]+=1
                    if st>0:post_bets[k,side]+=1
                if both_live and last>=0 and last!=partner and call>0:
                    x[k,133]+=1
                    x[k,134]+=(fo and strong[a])
                    x[k,135]+=(ca and not strong[a] and not draw[a])*amt/bb
                    x[k,136]+=ag
                    x[k,137]+=fo*odds
                    x[k,131]+=(ca and strength[a]<.4 and strong[partner])
                    x[k,132]+=(fo and strong[a] and ranks[partner]>ranks[a])
                hu=both_live and ac['active']==2
                if hu:
                    x[k,34]+=1;x[k,35]+=ch;x[k,36]+=ca;x[k,37]+=ag
                    x[k,53]+=abs(strength[a]-strength[partner])
                    # Checking against an all-in player is not a soft-play opportunity.
                    opportunity=stack>0 and rem[partner]>0
                    x[k,69]+=opportunity
                    if not opportunity and pas:x[k,149]+=1
                    if opportunity:
                        hu_seen[k,st]=1
                        x[k,slot+4]+=ch;x[k,slot+5]+=ca;x[k,slot+6]+=ag
                        street_checks[k,st,side]+=ch
                        if st>0:
                            x[k,124]+=(ch and strong[a] and not draw[a] and not paired and not monotone)
                            x[k,125]+=(ch and draw[a])
                            x[k,126]+=(ch and ranks[a]==ranks[partner])
                        x[k,38]+=(strong[a] and pas)
                        passive[k,side]+=pas
                if both_live and not hu:
                    if ag and last!=partner:
                        member_out[k,side]+=1;x[k,44]+=1
                    x[k,45]+=(fo and strong[a] and ranks[partner]>ranks[a] and last!=partner and last>=0)
                    x[k,46]+=(ag and strength[a]<0.4 and strong[partner])
                    x[k,66]+=(fo and strong[partner] and last!=partner)
            own_street_raises[a,st]+=ag;own_street_calls[a,st]+=ca
            comm[a]+=amt;rem[a]=max(0.,stack-amt)
            if ag:last=a
            if fo:live[a]=False
        for k in range(nk):
            s1=seat_a[k];s2=seat_b[k]
            x[k,39]=min(passive[k,0],passive[k,1])
            x[k,43]=min(member_out[k,0],member_out[k,1])
            x[k,60]=min(member_agg[s1],member_agg[s2]);x[k,61]=abs(member_agg[s1]-member_agg[s2])
            x[k,62]=min(member_calls[s1],member_calls[s2]);x[k,63]=abs(member_out[k,0]-member_out[k,1])
            x[k,52]=min(cat[s1],cat[s2]);x[k,53]/=max(1.,x[k,34])
            for street_i in range(4):
                x[k,127]+=min(street_checks[k,street_i,0],street_checks[k,street_i,1])
                x[k,128]+=(street_bets[k,street_i,0]>0 and street_bets[k,street_i,1]>0)
                x[k,146]+=hu_seen[k,street_i]
                x[k,147]+=(hu_seen[k,street_i]>0 and x[k,77+10*street_i+6]==0)
            x[k,130]=min(post_bets[k,0],post_bets[k,1])
            for street_i in range(4):
                pa=pair_out_street[k,street_i,0];pb_=pair_out_street[k,street_i,1]
                x[k,EXTRA_START+28]+=((pa>0)!=(pb_>0))
                x[k,EXTRA_START+29]+=((pa>0) and (pb_>0))
            # Per-hand board texture is a post-hoc observed context, not a future action feature.
            x[k,140]=paired
            x[k,141]=monotone
            # Interpretable hypotheses, not generator signatures or solved poker EV.
            x[k,70]=math.log1p(x[k,5])*(0.25+0.35*x[k,29]+0.9*x[k,32])+math.log1p(x[k,47]+2*x[k,48])+0.3*math.log1p(x[k,33])
            x[k,71]=((x[k,81]+x[k,91]+x[k,101]+x[k,111]+.7*(x[k,82]+x[k,92]+x[k,102]+x[k,112]))+1.5*x[k,38]+x[k,39])/(1.+1.5*x[k,37])+0.4*x[k,32]
            x[k,72]=math.log1p(x[k,40]+2*x[k,43]+0.35*x[k,44])/(1.+0.35*x[k,30])+0.3*x[k,43]
            x[k,73]=(1.6*x[k,45]+x[k,46]+0.2*x[k,66]+0.5*x[k,32])/(1.+0.2*x[k,16])+0.25*math.log1p(x[k,48])
            x[k,74]=max(x[k,70],x[k,71],x[k,72],x[k,73])
        # Compute the contrast AFTER all 15 rule values are available.
        for k in range(nk):
            for qi in range(len(CONTRAST_IDX)):
                val=max(0.,x[k,CONTRAST_IDX[qi]]);nless=0.;neq=0.;total=0.;logs=0.;maxother=0.
                for kk in range(nk):
                    vv=max(0.,x[kk,CONTRAST_IDX[qi]])
                    nless+=vv<val;neq+=vv==val;total+=vv
                    if kk!=k:logs+=math.log1p(vv);maxother=max(maxother,vv)
                base=CONTRAST_START+qi*4
                x[k,base]=(nless+.5*neq)/max(1,nk)
                x[k,base+1]=val/(1.+total)
                x[k,base+2]=math.log1p(val)-logs/max(1,nk-1)
                x[k,base+3]=(val-maxother)/(1.+val+maxother)
            if pidx[k]>=0:
                out[cursor]=x[k];out_p[cursor]=pidx[k];out_h[cursor]=h;cursor+=1
    return out[:cursor],out_p[:cursor],out_h[:cursor],total_prof,shared_prof



--- CODE CELL 12 ---
def build_pairs(pack, cfg):
    pi=pd.Index(pack.player_ids)
    rows=[]; rng=np.random.default_rng(cfg.seed)
    known_keys=set(); pos_players=set()
    for r in pack.labels.itertuples(index=False):
        u,v=sorted(pi.get_indexer([r.player_1,r.player_2]).tolist())
        assert u>=0 and v>=0 and u!=v
        assert pack.player_table[u]==pack.player_table[v]
        t=int(pack.player_table[u]);a=int(pack.player_local[u]);b=int(pack.player_local[v])
        n=int(pack.pair_counts[t,0,a,b]); assert n>0
        rows.append(dict(pair_id=str(r.pair_id),u=u,v=v,pool=t,phase=0,label=int(r.label),
                         family=str(r.behavior_family),shared_hands=n))
        known_keys.add((u,v))
        if r.label==1:pos_players.update((u,v))
    # Draws are explicit unknowns, never relabeled confirmed negatives.
    for t in range(len(pack.table_ids)):
        members=np.flatnonzero(pack.player_table==t); opts=[]
        for u,v in combinations(members.tolist(),2):
            if (u,v) in known_keys or u in pos_players or v in pos_players:continue
            a=pack.player_local[u];b=pack.player_local[v];n=int(pack.pair_counts[t,0,a,b])
            if n>=cfg.unknown_min_shared:opts.append((u,v,n))
        if opts:
            for j in rng.permutation(len(opts))[:cfg.unknown_per_pool]:
                u,v,n=opts[j]
                rows.append(dict(pair_id=f'UNKNOWN_{t}_{u}_{v}',u=u,v=v,pool=t,phase=0,label=-1,family='unknown',shared_hands=n))
    for r in pack.eval_pairs.itertuples(index=False):
        u,v=sorted(pi.get_indexer([r.player_1,r.player_2]).tolist())
        assert u>=0 and v>=0 and u!=v and pack.player_table[u]==pack.player_table[v]
        t=int(pack.player_table[u]);a=int(pack.player_local[u]);b=int(pack.player_local[v]);n=int(pack.pair_counts[t,1,a,b])
        if hasattr(r,'shared_hands'):assert n==int(r.shared_hands),'Evaluation shared-hand mismatch'
        rows.append(dict(pair_id=str(r.pair_id),u=u,v=v,pool=t,phase=1,label=-1,family='unknown',shared_hands=n))
    pairs=pd.DataFrame(rows).reset_index(drop=True);pairs['pid']=np.arange(len(pairs),dtype=np.int32)
    assert not pairs.duplicated(['phase','u','v']).any()
    assert not pairs.duplicated('pair_id').any()
    if not pack.fixture:
        positive_ids=set(pack.player_ids[list(pos_players)])
        assert not set(pack.eval_pairs.player_1)&positive_ids and not set(pack.eval_pairs.player_2)&positive_ids
    pairs.to_csv(Path(cfg.output_dir)/'pair_manifest.csv',index=False)
    log('pair_manifest',development=int((pairs.phase==0).sum()),unknown=int(((pairs.phase==0)&(pairs.label<0)).sum()),evaluation=int((pairs.phase==1).sum()))
    return pairs

PROFILE_NAMES=[f'ctx_{event}_{stat}' for event in ['aggression','fold','call','check']
               for stat in ['resid_sum','resid_max','resid_min','delta_sum','delta_gap','outside_support']]
TEMP_NAMES=[f'{r}_{stat}' for r in RULES for stat in ['peak10_z','peak25_z','late_early_gap','top5_span','nonzero_rate']]
PAIR_FEATURES=([f'{c}_{stat}' for stat in ['mean','max','p95','top5'] for c in MODEL_HAND]
               + ['log_shared','flow_direction_imbalance','flow_signed_abs_mean']+PROFILE_NAMES+TEMP_NAMES
               + [f'{s}_{stat}' for s in ['transfer_action','surrender_action','pressure_action'] for stat in ['coherence','mass_per_hand','top5_coherence']])


def contextual_profile_features(local_pair, u_local, v_local, total, shared):
    result=[]
    for event in range(1,5):
        zs=[];ds=[];support=[]
        for side,player in enumerate((u_local,v_local)):
            src=shared[local_pair,side]
            outside=np.maximum(total[player]-src,0.)
            pooled=total.sum(axis=0)
            prior=(pooled[:,event]+1)/(pooled[:,0]+4.)
            prob=(outside[:,event]+12*prior)/(outside[:,0]+12.)
            n=src[:,0]; residual=src[:,event]-n*prob
            zs.append(float(np.clip(residual.sum()/np.sqrt((n*prob*(1-prob)).sum()+4),-20,20)))
            ds.append(float(residual.sum()/(n.sum()+10)))
            support.append(float(np.log1p(outside[:,0].sum())))
        result.extend([sum(zs),max(zs),min(zs),sum(ds),abs(ds[0]-ds[1]),min(support)])
    return result


def aggregate_one(x, profile):
    n=len(x)
    if n==0:return np.zeros(len(PAIR_FEATURES),np.float32)
    a=x[:,MODEL_HAND_IDX];k=min(5,n)
    top=np.partition(a,n-k,axis=0)[-k:].mean(axis=0)
    result=[*a.mean(axis=0),*a.max(axis=0),*np.quantile(a,.95,axis=0),*top]
    flow=x[:,HF['flow_signed']]
    result += [math.log1p(n),abs(float(flow.sum()))/(float(np.abs(flow).sum())+1),abs(float(flow.mean()))]
    result += list(profile)
    prog=x[:,HF['progress']]
    for col in RULE_IDX:
        z=x[:,col].astype(np.float64);mu=z.mean();sd=z.std()
        for w in (10,25):
            w=min(w,n);cs=np.r_[0,np.cumsum(z)];peak=np.max((cs[w:]-cs[:-w])/w)
            # Overlap / unequal exposure makes this a descriptive burst feature, not a p-value.
            result.append(float(np.clip((peak-mu)/(sd/np.sqrt(w)+.15),0,30)))
        early=z[prog<.5];late=z[prog>=.5]
        result.append(float(late.mean()-early.mean()) if len(early) and len(late) else 0.)
        ix=np.argsort(-z,kind='stable')[:min(5,n)]
        result.append(float(np.ptp(prog[ix])) if len(ix)>1 else 0.)
        result.append(float(np.mean(z>0)))
    for name in ['transfer_action','surrender_action','pressure_action']:
        signed=x[:,HF[name+'_signed']].astype(np.float64);mass=np.abs(signed)
        topids=np.argsort(-mass,kind='stable')[:min(5,n)]
        result += [abs(signed.sum())/(1+mass.sum()),mass.mean(),abs(signed[topids].sum())/(1+mass[topids].sum())]
    arr=np.asarray(result,np.float32)
    assert len(arr)==len(PAIR_FEATURES)
    return np.nan_to_num(arr,nan=0,posinf=0,neginf=0)




--- CODE CELL 14 ---
@njit(cache=False)
def action_null_context(strength, street, to_call, pot, stack, active):
    odds=min(to_call,stack)/(max(pot,1e-12)+min(to_call,stack)+1e-12)
    ob=0 if to_call<=0 else (1 if odds<=.10 else (2 if odds<=.25 else (3 if odds<=.40 else 4)))
    sb=min(7,max(0,int(strength*8)))
    return ((street*8+sb)*5+ob)*2+int(active>2)

@njit(cache=False)
def reference_distribution(total, shared_actor, pooled, shared_pair, context):
    # No labels and no partner cards. All of the actor's shared-hand decisions are subtracted.
    outside=np.maximum(total[context]-shared_actor[context],0.)
    remain=np.maximum(pooled[context]-shared_pair[0,context]-shared_pair[1,context],0.)
    prior=np.empty(4,np.float64);prob=np.empty(4,np.float64)
    for e in range(4):
        prior[e]=(remain[1+e]+.5)/(remain[0]+2.)
        prob[e]=(outside[1+e]+16.*prior[e])/(outside[0]+16.)
    # All count updates are complete-event updates; normalization also guards roundoff.
    prob/=prob.sum()
    return prob,outside,remain

@njit(cache=False)
def action_null_features(hids, sp, sc, sv, board, hv, aa, ptr, plocal,
                         pairmap, lows, highs, sorted_hids, row_lookup, nrows):
    npairs=len(lows);m=pairmap.shape[0];nctx=NULL_CONTEXTS
    total=np.zeros((m,nctx,NULL_FIELDS),np.float32)
    shared=np.zeros((npairs,2,nctx,NULL_FIELDS),np.float32)
    nact=0
    for h in hids:nact+=ptr[h+1]-ptr[h]
    # event = output row, pair, actor, side, context, action, size, role,
    #         centered partner strength, benefit, own strong, own weak,
    #         reference exposed street, role eligible. Buffers are bounded by one pool/phase.
    event=np.zeros((nact*5,12),np.float32);ne=0
    forced=np.zeros(nrows,np.float32)
    for h in hids:
        live=np.ones(6,np.bool_);rem=sv[h,:,0].copy();button=int(hv[h,2]);bb=max(hv[h,0],1e-12)
        rem[(button+1)%6]-=min(rem[(button+1)%6],hv[h,1])
        rem[(button+2)%6]-=min(rem[(button+2)%6],bb)
        strength=np.zeros(6,np.float64);ranks=np.zeros(6,np.int64)
        strong=np.zeros(6,np.bool_);draw=np.zeros(6,np.bool_)
        prev=-1;last=-1;hp=np.searchsorted(sorted_hids,h)
        for ai in range(ptr[h],ptr[h+1]):
            ac=aa[ai];a=-1
            for s in range(6):
                if sp[h,s]==ac['player']:a=s;break
            if a<0:raise ValueError('Unseated actor in null replay')
            st=int(ac['street']);k=int(ac['kind']);amt=float(ac['amount'])
            call=float(ac['call']);pot=max(float(ac['pot']),bb);stack=max(float(ac['stack']),0.)
            if st!=prev:
                nb=0 if st==0 else st+2
                for s in range(6):
                    strength[s],ranks[s],cat_,strong[s],draw[s]=hand_context(sc[h,s,0],sc[h,s,1],board[h],nb)
                prev=st;last=-1
            ag=k==3 or k==4 or (k==5 and amt>call+1e-6)
            et=3 if ag else (0 if k==0 else (1 if k==1 else 2))
            ctx=action_null_context(strength[a],st,call,pot,stack,int(ac['active']))
            loc=plocal[sp[h,a]];size=np.log1p(amt/pot)
            bettable_others=0
            for o in range(6):
                if o!=a and live[o] and rem[o]>0:bettable_others+=1
            reference_eligible=stack>0 and not (et==1 and int(ac['active'])==2 and bettable_others==0)
            if reference_eligible:
                total[loc,ctx,0]+=1;total[loc,ctx,1+et]+=1
                total[loc,ctx,5+et]+=size;total[loc,ctx,9+et]+=size*size
            for b in range(6):
                if a==b:continue
                p=pairmap[loc,plocal[sp[h,b]]]
                if p<0:continue
                side=0 if sp[h,a]==lows[p] else 1
                if reference_eligible:
                    shared[p,side,ctx,0]+=1;shared[p,side,ctx,1+et]+=1
                    shared[p,side,ctx,5+et]+=size;shared[p,side,ctx,9+et]+=size*size
                if not live[a] or not live[b]:continue
                row=row_lookup[hp,p]
                if row<0:raise ValueError('Pair-hand mapping missing in null replay')
                hu=int(ac['active'])==2
                direct=(call>0 and last==b) or hu
                if not reference_eligible:
                    forced[row]+=1
                    continue
                role=0 if direct else 1
                other_sum=0.;other_n=0
                for o in range(6):
                    if o!=a and o!=b and live[o]:other_sum+=strength[o];other_n+=1
                center=strength[b]-(other_sum/other_n if other_n>0 else .5)
                ahead=st>0 and ranks[a]>ranks[b];behind=st>0 and ranks[a]<ranks[b]
                weak=strength[a]<.40 and not draw[a]
                benefit=0.
                if direct:
                    if et==0 and (ahead or strong[a]):benefit=1.
                    if et==2 and (behind or weak):benefit=1.
                    if et==1 and strong[a] and hu and rem[b]>0:benefit=.8
                    if et==3 and behind and weak:benefit=.6
                else:
                    if et==0 and strong[a] and ranks[b]>ranks[a]:benefit=1.
                    if et==3 and weak and strong[b]:benefit=.8
                    if et==2 and weak and strong[b]:benefit=.4
                event[ne,0]=row;event[ne,1]=p;event[ne,2]=loc;event[ne,3]=side
                event[ne,4]=ctx;event[ne,5]=et;event[ne,6]=size;event[ne,7]=role
                event[ne,8]=center;event[ne,9]=benefit;event[ne,10]=strong[a];event[ne,11]=weak
                ne+=1
            rem[a]=max(0.,stack-amt)
            if et==3:last=a
            if et==0:live[a]=False
    pooled=total.sum(axis=0)
    result=np.zeros((nrows,N_NULL_FEATURES),np.float32)
    for i in range(ne):
        ev=event[i];row=int(ev[0]);p=int(ev[1]);actor=int(ev[2]);side=int(ev[3]);ctx=int(ev[4]);et=int(ev[5]);role=int(ev[7]);off=17*role
        prob,out,remain=reference_distribution(total[actor],shared[p,side],pooled,shared[p],ctx)
        pr=max(1e-5,prob[et]);surprise=min(11.,-np.log(pr));benefit=float(ev[9])
        result[row,off]+=1;result[row,off+1]+=surprise
        result[row,off+2]=max(result[row,off+2],surprise)
        result[row,off+3]+=pr<.08
        for e in range(4):result[row,off+4+e]+=(1. if et==e else 0.)-prob[e]
        support=np.log1p(out[0]);result[row,off+15]+=support
        if result[row,off]==1:result[row,off+16]=pr
        else:result[row,off+16]=min(result[row,off+16],pr)
        if et>=2:
            # Sizing expectation is conditional on actual event and the same own-card context.
            mu0=remain[5+et]/max(1.,remain[1+et]);m20=remain[9+et]/max(1.,remain[1+et])
            mu=(out[5+et]+8.*mu0)/(out[1+et]+8.)
            m2=(out[9+et]+8.*m20)/(out[1+et]+8.)
            zz=min(12.,abs(ev[6]-mu)/np.sqrt(max(.025,m2-mu*mu)))
            result[row,off+8]=max(result[row,off+8],zz);result[row,off+9]+=zz
        result[row,off+10]+=((1. if et==3 else 0.)-prob[3])*ev[8]
        result[row,off+11]+=((1. if et==0 else 0.)-prob[0])*ev[8]
        result[row,off+12]+=surprise*benefit
        result[row,off+13]+=surprise*(et==0)*ev[10]
        result[row,off+14]+=surprise*(et==3)*ev[11]
        result[row,34]=max(result[row,34],surprise*benefit)
        result[row,35]=max(result[row,35],surprise)
        result[row,40]+=(pr<.08 and benefit>.5)
    for row in range(nrows):
        for role in range(2):
            off=17*role;n=max(1.,result[row,off]);result[row,off+15]/=n
        result[row,36]=abs(result[row,10]+result[row,27])+abs(result[row,11]+result[row,28])
        result[row,37]=result[row,0];result[row,38]=result[row,17]
        result[row,39]=result[row,34]*(1.+.25*np.log1p(result[row,40]))+.25*result[row,36]
        result[row,41]=forced[row]
    return result


def augment_null_block(pack,hids,x,local_pid,hid,pairmap,us,vs):
    sorted_hids=np.sort(np.asarray(hids,np.int32));rows=np.full((len(hids),len(us)),-1,np.int32)
    if len(hid):rows[np.searchsorted(sorted_hids,hid),local_pid]=np.arange(len(hid),dtype=np.int32)
    z=action_null_features(hids,pack.seat_players,pack.seat_cards,pack.seat_values,pack.board,
        pack.hand_values,pack.actions,pack.action_ptr,pack.player_local,pairmap,us,vs,sorted_hids,rows,len(x))
    assert z.shape==(len(x),len(NULL_FEATURES)) and np.isfinite(z).all()
    x[:,NULL_START:NULL_START+len(NULL_FEATURES)]=z
    return x




--- CODE CELL 16 ---
def extract_features(pack,pairs,cfg):
    root=pack.root/'features';root.mkdir(exist_ok=True)
    fp=root/'pair_features.npy';marker=root/'COMPLETE.json'
    if marker.exists() and fp.exists() and not pack.fixture:
        z=np.load(fp,mmap_mode='r')
        if z.shape==(len(pairs),len(PAIR_FEATURES)):
            log('feature_cache_reused',shape=z.shape)
            return z,sorted(root.glob('block_*.npz'))
    px=np.zeros((len(pairs),len(PAIR_FEATURES)),np.float32);paths=[]
    hpools=pack.hand_values[:,6].astype(np.int32);hphase=pack.hand_values[:,5].astype(np.int8)
    for t in range(len(pack.table_ids)):
        for phase in (0,1):
            sub=pairs[(pairs.pool==t)&(pairs.phase==phase)]
            if sub.empty:continue
            ids=sub.pid.to_numpy(np.int32);us=sub.u.to_numpy(np.int32);vs=sub.v.to_numpy(np.int32)
            mp=np.full((pack.max_pool,pack.max_pool),-1,np.int32)
            for j,(u,v) in enumerate(zip(us,vs)):
                a=pack.player_local[u];b=pack.player_local[v];mp[a,b]=mp[b,a]=j
            hh=np.flatnonzero((hpools==t)&(hphase==phase))
            hh=hh[np.argsort(pack.times[hh],kind='stable')].astype(np.int32)
            path=root/f'block_{phase}_{t:04d}.npz'
            if path.exists() and not pack.fixture:
                z=np.load(path); x=z['x'];lp=z['local_pid'];h=z['hid'];tp=z['total_prof'];sp=z['shared_prof']
                assert np.array_equal(z['pids'],ids)
            else:
                x,lp,h,tp,sp=replay_block(hh,pack.seat_players,pack.seat_cards,pack.seat_values,pack.board,
                                         pack.hand_values,pack.times,pack.actions,pack.action_ptr,
                                         pack.player_local,mp,us,vs)
                # This sort is solely to group rows, preserving real hand chronology inside each pair.
                order=np.argsort(lp,kind='stable');x=x[order];lp=lp[order];h=h[order]
                x=augment_null_block(pack,hh,x,lp,h,mp,us,vs)
                tmp=path.with_name(path.stem+'.tmp.npz')
                np.savez_compressed(tmp,x=x,local_pid=lp,hid=h,pids=ids,total_prof=tp,shared_prof=sp)
                tmp.replace(path)
            starts=np.r_[0,np.cumsum(np.bincount(lp,minlength=len(ids)))]
            for j,pid in enumerate(ids):
                xx=x[starts[j]:starts[j+1]]
                assert len(xx)==pairs.loc[pid,'shared_hands']
                profile=contextual_profile_features(j,pack.player_local[us[j]],pack.player_local[vs[j]],tp,sp)
                px[pid]=aggregate_one(xx,profile)
            paths.append(path)
            del x,lp,h,tp,sp
        if (t+1)%20==0 or t+1==len(pack.table_ids):log('gameplay_features',pools_complete=t+1,pools_total=len(pack.table_ids))
    np.save(fp,px)
    save_json(root/'feature_schema.json',{'hand':HAND_FEATURES,'pair':PAIR_FEATURES,'excluded':['IDs','account metadata','absolute timestamps','raw progress','signed orientation']})
    save_json(marker,{'shape':px.shape,'feature_count':len(PAIR_FEATURES)})
    return px,paths



--- CODE CELL 18 ---
def evidence_gold_map(pack,pairs):
    lookup=dict(zip(pairs.pair_id,pairs.pid)); hi=pd.Index(pack.hand_ids)
    gold=defaultdict(set)
    required={'pair_id','hand_id'}
    if not required.issubset(pack.evidence.columns):raise ValueError('Evidence CSV needs pair_id and hand_id')
    for r in pack.evidence.itertuples(index=False):
        if r.pair_id not in lookup:raise ValueError('Evidence pair absent from manifest')
        h=int(hi.get_indexer([r.hand_id])[0]);p=int(lookup[r.pair_id])
        assert h>=0 and pack.hand_values[h,5]==0 and pairs.loc[p,'label']==1
        assert pairs.loc[p,'u'] in pack.seat_players[h] and pairs.loc[p,'v'] in pack.seat_players[h]
        gold[p].add(h)
    assert all(len(gold[p])>0 for p in pairs.loc[(pairs.phase==0)&(pairs.label==1),'pid'])
    return dict(gold)

def assign_folds(pairs,cfg):
    dev=np.flatnonzero(pairs.phase.to_numpy()==0)
    strata=np.where(pairs.loc[dev,'label'].to_numpy()==1,pairs.loc[dev,'family'].to_numpy(),
                    np.where(pairs.loc[dev,'label'].to_numpy()==0,'confirmed_none','unknown'))
    splitter=StratifiedGroupKFold(n_splits=cfg.n_folds,shuffle=True,random_state=cfg.seed)
    folds=np.full(len(pairs),-1,np.int8)
    for f,(_,va) in enumerate(splitter.split(np.zeros((len(dev),1)),strata,pairs.loc[dev,'pool'])):
        folds[dev[va]]=f
    assert (folds[dev]>=0).all()
    for f in range(cfg.n_folds):
        tr=pairs.loc[(folds>=0)&(folds!=f)]
        va=pairs.loc[folds==f]
        assert set(tr.pool).isdisjoint(set(va.pool))
        assert set(tr.u)|set(tr.v)  # Ensure nonempty; disjointness checked explicitly below.
        assert (set(tr.u)|set(tr.v)).isdisjoint(set(va.u)|set(va.v))
    return folds

def ranked_ap(y,score,ids):
    y=np.asarray(y,dtype=np.int8)
    if y.sum()==0:return 0.
    order=np.lexsort((np.asarray(ids,dtype=str),-np.asarray(score,dtype=float)))
    rel=y[order];return float(np.sum(np.cumsum(rel)/np.arange(1,len(rel)+1)*rel)/rel.sum())

def ap5(hands,gold):
    if not gold:return 0.
    seen=set();hits=0;total=0.;rank=0
    for h in hands[:5]:
        if h=='NO_EVIDENCE':continue
        if h in seen:raise ValueError('Repeated evidence ID')
        seen.add(h);rank+=1
        if h in gold:hits+=1;total+=hits/rank
    return total/min(5,len(gold))

def metric_proxy(pair_frame,risk,behavior,evidence_lists,gold):
    mask=pair_frame.label.to_numpy()>=0
    df=pair_frame.loc[mask];r=np.asarray(risk)[mask];b=np.asarray(behavior)[mask]
    y=df.label.to_numpy();ids=df.pair_id.to_numpy()
    pap=ranked_ap(y,r,ids)
    bmap=np.mean([ranked_ap((df.family.to_numpy()==fam).astype(int),np.where(b==fam,r,0.),ids) for fam in FAMILIES])
    ev=[]
    for p in df.loc[df.label==1,'pid']:
        ev.append(ap5(evidence_lists.get(int(p),[]),gold.get(int(p),set())))
    emap=float(np.mean(ev)) if ev else 0.
    return dict(pair_ap=pap,evidence_map5=emap,behavior_map=float(bmap),
                composite_proxy=.70*pap+.20*emap+.10*float(bmap),
                sklearn_pair_ap=float(average_precision_score(y,r)) if y.sum() else 0.,
                n_known=len(df),n_positive=int(y.sum()),metric_status='PUBLIC_SPEC_PROXY_NOT_OFFICIAL',
                label_population='CONFIRMED_LABELS_ONLY; not an unbiased estimate of hidden evaluation prevalence')



--- CODE CELL 20 ---
# Full shared-hand scoring. Positive-pair unannotated hands are weak retrieval background,
# NOT claims of benign gameplay. No unknown-pair labels enter evidence supervision.
EVIDENCE_FEATURES=MODEL_HAND+[f'{c}_within_pair_pct' for c in RULES]+[
    'partner_fold_rate','partner_call_rate','partner_raise_rate','hu_eligible_passive_rate',
    'outsider_fold_rate','duo_pressure_rate','call_over_pot','transfer_over_pot',
    'live_partner_action_rate','contribution_over_stack','partner_call_over_stack','eligible_hu_rate']
ATTRIBUTION_RATIOS = ['directed_local_attribution','soft_local_attribution','isolation_local_attribution','other_local_attribution','committed_fold_per_partner_face','outsider_pair_pressure_fraction','partner_river_waste_pot_ratio','live_partner_versus_outsider_rate']
EVIDENCE_FEATURES += ATTRIBUTION_RATIOS
for name in RULES:
    EVIDENCE_FEATURES += [f'{name}_neighbor5',f'{name}_neighbor13',f'{name}_robust_deviation']

def percentiles(v):
    from scipy.stats import rankdata
    return (rankdata(v,method='average')/max(1,len(v))).astype(np.float32)

def local_mean(v,width):
    n=len(v)
    if n==0:return np.empty(0,np.float32)
    j=np.arange(n);a=np.maximum(0,j-width//2);b=np.minimum(n,j+width//2+1)
    cs=np.r_[0,np.cumsum(np.asarray(v,np.float64))]
    return ((cs[b]-cs[a])/np.maximum(1,b-a)).astype(np.float32)

def anchor_evidence_matrix(x):
    if not len(x):return np.empty((0,len(ANCHOR_EVIDENCE_FEATURES)),np.float32)
    c=lambda name:x[:,HF[name]]
    extra=[percentiles(x[:,i]) for i in RULE_IDX]
    denom=c('partner_facing')+1
    extra += [c('partner_folds')/denom,c('partner_calls')/denom,c('partner_raises')/denom,
        (c('hu_checks')+c('hu_calls'))/(c('hu_opportunity_stacks_live')+1),
        c('outsider_folds_livepair')/(c('member_aggr_to_outsiders')+1),
        c('duo_pressure')/(c('aggressive_actions')+1),c('partner_call_bb')/(c('pot_bb')+1),
        c('transfer_proxy_bb')/(c('pot_bb')+1),c('partner_live_actions')/(c('n_actions')+1),
        c('pair_contrib_bb')/(c('min_stack_bb')+c('max_stack_bb')+1),
        c('partner_call_bb')/(c('min_stack_bb')+1),c('hu_opportunity_stacks_live')/(c('hu_actions')+1)]
    extra += [c(name+'_peer_share')*np.log1p(np.maximum(c(name),0)) for name in RULES]
    extra += [c('partner_small_raise_fold_committed_bb')/(c('partner_facing')+1),
              c('reciprocal_outsider_pressure_streets')/(c('member_aggr_to_outsiders')+1),
              (c('partner_river_ahead_fold_pot_bb')+c('partner_river_losing_raise_bb'))/(c('pot_bb')+1),
              c('voluntary_vs_partner_action_count')/(c('voluntary_vs_outsider_action_count')+1)]
    for i in RULE_IDX:
        v=x[:,i];med=np.median(v);mad=np.median(abs(v-med))+.25
        extra.extend([local_mean(v,5),local_mean(v,13),np.clip((v-med)/mad,-12,12)])
    z=np.column_stack([x[:,MODEL_HAND_IDX],*extra]).astype(np.float32)
    assert z.shape[1]==len(ANCHOR_EVIDENCE_FEATURES)
    return np.nan_to_num(z,nan=0,posinf=0,neginf=0)


ANCHOR_EVIDENCE_FEATURES = list(EVIDENCE_FEATURES)
QUERY_SOURCES = [
    'rule_directed','rule_soft','rule_isolation','rule_other',
    'partner_call_bb','partner_strong_fold','partner_weaker_call_bb','hu_strong_passive',
    'duo_pressure','member_aggr_to_outsiders','strong_fold_to_outsider_partner_better',
    'weak_aggr_partner_strong','river_losing_call_bb','river_winning_fold_bb',
    'partner_raise_then_fold','partner_river_ahead_fold_pot_bb','partner_river_losing_raise_bb',
    'partner_low_odds_ahead_fold','hu_river_best_check','hu_strong_check_nonallin',
    'outsider_folds_after_duo','reciprocal_outsider_pressure_streets',
    'null_partner_surprisal_sum','null_partner_benefit_surprise','null_partner_sizing_z_max',
    'null_outsider_benefit_surprise','null_dependency_abs','null_action_score',
    'partner_call_commit_max','partner_fold_committed_max','voluntary_vs_partner_action_count',
    'voluntary_vs_outsider_action_count']
QUERY_FEATURES = [f'query_{name}_{stat}' for name in QUERY_SOURCES
                  for stat in ('positive_percentile','robust_excess','signal_share')]
QUERY_FEATURES += ['query_log_exposure','query_hu_opportunity_rate','query_partner_facing_rate',
                   'query_pair_contribution_median','query_action_null_median']
EVIDENCE_FEATURES = ANCHOR_EVIDENCE_FEATURES + QUERY_FEATURES

def evidence_matrix(x):
    """First 289 columns are the V4 matrix. Add label-free, query-specific contrasts."""
    base=anchor_evidence_matrix(x)
    if not len(x):return np.empty((0,(len(ANCHOR_EVIDENCE_FEATURES)+len(QUERY_FEATURES))),np.float32)
    more=[]
    for name in QUERY_SOURCES:
        v=np.maximum(0.,x[:,HF[name]].astype(np.float64))
        active=v>0; pct=np.zeros(len(v),np.float32)
        if active.any():pct[active]=percentiles(v[active])
        z=np.log1p(v);med=np.median(z);mad=np.median(np.abs(z-med))
        more.extend([pct,np.clip((z-med)/(mad+.15),-12.,12.),v/(v.sum()+1.)])
    constants=[np.log1p(len(x)),np.mean(x[:,HF['hu_opportunity_stacks_live']]>0),
               np.mean(x[:,HF['partner_facing']]>0),np.log1p(np.median(x[:,HF['pair_contrib_bb']])),
               np.log1p(np.median(np.maximum(0,x[:,HF['null_action_score']])))]
    more.extend([np.full(len(x),v,np.float32) for v in constants])
    out=np.column_stack([base,*more]).astype(np.float32)
    assert out.shape[1]==(len(ANCHOR_EVIDENCE_FEATURES)+len(QUERY_FEATURES))
    return np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)


def group_ranges(local_pid,n):
    return np.r_[0,np.cumsum(np.bincount(local_pid,minlength=n))]

def load_evidence_training(paths,pairs,gold):
    rows=[]
    for path in paths:
        if not path.name.startswith('block_0_'):continue
        with np.load(path) as z:
            pids=z['pids'];x=z['x'];hs=z['hid'];starts=group_ranges(z['local_pid'],len(pids))
            for j,p in enumerate(pids):
                label=int(pairs.loc[p,'label'])
                if label<0:continue
                a,b=starts[j:j+2];xx=x[a:b];hh=hs[a:b]
                y=np.array([int(h) in gold.get(int(p),set()) for h in hh],np.int8)
                rows.append((p,evidence_matrix(xx),y,hh))
    if not rows:raise ValueError('No evidence training pairs.')
    pp=np.concatenate([np.full(len(v[1]),v[0],np.int32) for v in rows])
    data=dict(pid=pp,x=np.concatenate([v[1] for v in rows]),y=np.concatenate([v[2] for v in rows]),
              hid=np.concatenate([v[3] for v in rows]))
    data['pool']=pairs.loc[pp,'pool'].to_numpy(np.int32)
    data['known_negative']=pairs.loc[pp,'label'].to_numpy()==0
    data['family']=np.array([int(np.flatnonzero(FAMILIES==f)[0]) if f in FAMILIES else -1 for f in pairs.loc[pp,'family']],np.int8)
    log('evidence_training',hands=len(pp),annotated=int(data['y'].sum()),features=data['x'].shape[1],unknowns_used_as_confirmed_negative=False)
    return data



--- CODE CELL 22 ---
def lgb_params(cfg,seed,trees):
    return dict(n_estimators=int(trees),learning_rate=.035,num_leaves=15,max_depth=-1,
                min_child_samples=24,reg_lambda=8.,reg_alpha=.1,colsample_bytree=.86,
                subsample=.90,subsample_freq=1,n_jobs=cfg.threads,verbosity=-1,
                random_state=int(seed),deterministic=True,force_col_wise=True)

@njit(cache=False)
def ap5_relevance_value(rel, k=5):
    m=int(np.sum(rel));denom=min(k,m)
    if denom==0:return 0.
    value=0.;hits=0
    for j in range(min(k,len(rel))):
        if rel[j]>0:hits+=1;value+=hits/(j+1)
    return value/denom

@njit(cache=False)
def ap5_swap_delta(sorted_y, pos_rank, neg_rank, k=5):
    """Exact absolute change in truncated AP for swapping one positive and one negative."""
    n=len(sorted_y);denom=min(k,int(np.sum(sorted_y)))
    if denom==0 or min(pos_rank,neg_rank)>=k:return 0.
    prefix=np.cumsum(sorted_y)
    if pos_rank<neg_rank:
        delta=prefix[pos_rank]/(pos_rank+1.)
        for t in range(pos_rank+1,min(neg_rank,k)):
            if sorted_y[t]>0:delta+=1./(t+1.)
        if neg_rank<k:delta-=prefix[neg_rank]/(neg_rank+1.)
    else:
        delta=(prefix[neg_rank]+1.)/(neg_rank+1.)
        for t in range(neg_rank+1,min(pos_rank,k)):
            if sorted_y[t]>0:delta+=1./(t+1.)
        if pos_rank<k:delta-=prefix[pos_rank]/(pos_rank+1.)
    return max(0.,delta/denom)

@njit(cache=False)
def ap5_lambdas(scores, labels, ptr, k=5):
    """LambdaMART-style surrogate weighted by exact AP@5 swap deltas; not a differentiable AP loss."""
    grad=np.zeros(len(scores),np.float64);hess=np.full(len(scores),1e-7,np.float64)
    for q in range(len(ptr)-1):
        a=ptr[q];b=ptr[q+1]
        if b-a<2:continue
        order=np.argsort(-scores[a:b],kind='mergesort')
        yy=labels[a:b][order].astype(np.int32)
        m=int(np.sum(yy));denom=min(k,m)
        if m==0 or m==b-a:continue
        prefix=np.cumsum(yy)
        pos=np.where(yy>0)[0];neg=np.where(yy==0)[0]
        for p in pos:
            for v in neg:
                if min(p,v)>=k:continue
                if p<v:
                    delta=prefix[p]/(p+1.)
                    for t in range(p+1,min(v,k)):
                        if yy[t]>0:delta+=1./(t+1.)
                    if v<k:delta-=prefix[v]/(v+1.)
                else:
                    delta=(prefix[v]+1.)/(v+1.)
                    for t in range(v+1,min(p,k)):
                        if yy[t]>0:delta+=1./(t+1.)
                    if p<k:delta-=prefix[p]/(p+1.)
                delta=max(0.,delta/denom)
                if delta<=0:continue
                ip=a+order[p];iv=a+order[v]
                diff=max(-40.,min(40.,scores[ip]-scores[iv]))
                rho=1./(1.+math.exp(diff));g=delta*rho;hh=delta*rho*(1.-rho)
                grad[ip]-=g;grad[iv]+=g;hess[ip]+=hh;hess[iv]+=hh
    return grad,hess

class AP5Objective:
    def __init__(self,groups):
        self.ptr=np.r_[0,np.cumsum(groups)].astype(np.int64)
    def __call__(self,predictions,dataset):
        return ap5_lambdas(np.asarray(predictions,np.float64),dataset.get_label().astype(np.int8),self.ptr,5)

def fit_score_reference(scores):
    z=np.log(np.clip(scores,1e-7,1-1e-7)/np.clip(1-scores,1e-7,1))
    if len(z)<10:return (0.,1.)
    median,tail=np.quantile(z,[.50,.995])
    return (float(tail),float(max(.7,(tail-median)/3.)))


def normalize_score(scores,reference):
    z=np.log(np.clip(scores,1e-7,1-1e-7)/np.clip(1-scores,1e-7,1))
    return expit(np.clip((z-reference[0])/reference[1],-15,15)).astype(np.float32)


class AnchorHandExperts:
    """Three family experts plus a generic action-evidence learner. Scores are reference normalized, NOT probabilities."""
    def __init__(self,cfg,seed):self.cfg=cfg;self.seed=int(seed)
    def fit(self,data,mask,pairs):
        use=mask[data['pid']];poolpids=np.unique(data['pid'][use & data['known_negative']])
        rng=np.random.default_rng(self.seed+41)
        hold=poolpids[rng.permutation(len(poolpids))[:max(1,len(poolpids)//5)]] if len(poolpids)>5 else np.array([],np.int32)
        calibration=use&np.isin(data['pid'],hold)
        training=use&~calibration
        self.reference_pair_ids=hold;self.fit_pair_ids=np.unique(data['pid'][training])
        assert set(hold).isdisjoint(set(self.fit_pair_ids))
        X=data['x'][training];y=data['y'][training];fam=data['family'][training]
        pp=data['pid'][training];negative=data['known_negative'][training]
        self.training_pools=set(pairs.loc[np.flatnonzero(mask),'pool'].astype(int))
        self.classifiers=[];self.rankers=[];self.ap_rankers=[];self.references=[]
        _,inv,counts=np.unique(pp,return_inverse=True,return_counts=True)
        qw=np.clip(np.median(counts)/counts[inv],.25,4.).astype(np.float32)
        XC=data['x'][calibration]
        for f in range(3):
            target=((y==1)&(fam==f)).astype(np.int8)
            if target.sum()<2:raise ValueError(f'Insufficient evidence for {FAMILIES[f]}')
            w=np.where(target==1,10.,np.where(negative,1.,np.where(y==1,.7,.15))).astype(np.float32)*np.sqrt(qw)
            classifier=lgb.LGBMClassifier(**lgb_params(self.cfg,self.seed+f,self.cfg.hand_trees))
            classifier.fit(X,target,sample_weight=w);self.classifiers.append(classifier)
            if len(XC):ref=fit_score_reference(classifier.booster_.predict(XC,num_threads=self.cfg.threads))
            else:ref=(0.,1.)
            self.references.append(ref)
            ix=np.flatnonzero(fam==f);ix=ix[np.argsort(pp[ix],kind='stable')]
            _,groups=np.unique(pp[ix],return_counts=True)
            par=lgb_params(self.cfg,self.seed+41+f,self.cfg.rank_trees)
            par.update(num_leaves=15,min_child_samples=18,reg_lambda=8.,objective='lambdarank',metric='map',
                lambdarank_truncation_level=8,label_gain=[0,1])
            ranker=lgb.LGBMRanker(**par);ranker.fit(X[ix],y[ix],group=groups);self.rankers.append(ranker)
            ds=lgb.Dataset(X[ix],label=y[ix],group=groups,free_raw_data=True)
            params=dict(objective=AP5Objective(groups),metric='None',learning_rate=.035,num_leaves=15,
                min_data_in_leaf=18,min_sum_hessian_in_leaf=1e-4,lambda_l2=.5,feature_fraction=.9,
                bagging_fraction=.9,bagging_freq=1,seed=self.seed+211+f,num_threads=self.cfg.threads,
                verbosity=-1,force_col_wise=True,deterministic=True)
            self.ap_rankers.append(lgb.train(params,ds,num_boost_round=int(self.cfg.ap5_trees)))
        # The generic learner is deliberately restricted to observable action context/residuals.
        self.generic_indices=np.array([i for i,n in enumerate(ANCHOR_EVIDENCE_FEATURES) if
            n.startswith(('null_','partner_','hu_','outsider_','river_','voluntary_'))
            or n in ('postflop_actions','live_partner_action_rate','call_over_pot')],np.int32)
        self.generic=lgb.LGBMClassifier(**{**lgb_params(self.cfg,self.seed+1701,self.cfg.generic_hand_trees),'num_leaves':11})
        gw=np.where(y==1,10.,np.where(negative,1.,.15)).astype(np.float32)*np.sqrt(qw)
        self.generic.fit(X[:,self.generic_indices],y,sample_weight=gw)
        self.generic_reference=fit_score_reference(self.generic.booster_.predict(XC[:,self.generic_indices],num_threads=self.cfg.threads)) if len(XC) else (0.,1.)
        return self
    def predict(self,X):
        if len(X)==0:return tuple(np.empty((0,3),np.float32) for _ in range(3))
        c=np.column_stack([normalize_score(m.booster_.predict(X,num_threads=self.cfg.threads),ref) for m,ref in zip(self.classifiers,self.references)])
        r=np.column_stack([m.booster_.predict(X,num_threads=self.cfg.threads) for m in self.rankers]).astype(np.float32)
        a=np.column_stack([m.predict(X,num_threads=self.cfg.threads) for m in self.ap_rankers]).astype(np.float32)
        return c,r,a
    def predict_likelihood(self,X):
        if len(X)==0:return np.empty((0,3),np.float32)
        return np.column_stack([normalize_score(m.booster_.predict(X,num_threads=self.cfg.threads),ref) for m,ref in zip(self.classifiers,self.references)])
    def predict_generic(self,X):
        if not len(X):return np.empty(0,np.float32)
        return normalize_score(self.generic.booster_.predict(X[:,self.generic_indices],num_threads=self.cfg.threads),self.generic_reference)


class HandExperts(AnchorHandExperts):
    """Protected V4 anchor plus independently trained query, MAP and witness experts."""
    def fit(self,data,mask,pairs):
        anchor_data=dict(data);anchor_data['x']=data['x'][:,:len(ANCHOR_EVIDENCE_FEATURES)]
        super().fit(anchor_data,mask,pairs)
        use=mask[data['pid']]
        reference=use&np.isin(data['pid'],self.reference_pair_ids)
        train=use&~reference
        X=data['x'][train];y=data['y'][train];fam=data['family'][train]
        pp=data['pid'][train];neg=data['known_negative'][train]
        XC=data['x'][reference]
        _,inv,count=np.unique(pp,return_inverse=True,return_counts=True)
        qw=np.clip(np.median(count)/count[inv],.25,4.).astype(np.float32)
        self.query_models=[];self.query_refs=[]
        for f in range(3):
            target=((y==1)&(fam==f)).astype(np.int8)
            # Annotation likelihood, not a claim that unannotated hands are benign.
            weight=np.where(target,8.,np.where(neg,1.,np.where(y,.7,.12)))*np.sqrt(qw)
            model=lgb.LGBMClassifier(**{**lgb_params(self.cfg,self.seed+7001+f,self.cfg.query_trees),
                'num_leaves':11,'reg_lambda':14.,'min_child_samples':28})
            model.fit(X,target,sample_weight=weight)
            self.query_models.append(model)
            self.query_refs.append(fit_score_reference(model.booster_.predict(XC,num_threads=self.cfg.threads)) if len(XC) else (0.,1.))
        # Pool the three families for more effective training queries. The hypothesis
        # is an explicit one-hot input, never a pair identity or a hidden true label at inference.
        import xgboost as xgb
        ix=np.flatnonzero(fam>=0);ix=ix[np.argsort(pp[ix],kind='stable')]
        _,groups=np.unique(pp[ix],return_counts=True)
        family_input=np.eye(3,dtype=np.float32)[fam[ix]]
        self.map_model=xgb.XGBRanker(objective='rank:map',eval_metric='map@5',
            n_estimators=self.cfg.map_trees,learning_rate=.035,max_depth=4,
            min_child_weight=3.,reg_lambda=16.,reg_alpha=.05,subsample=.85,colsample_bytree=.85,
            tree_method='hist',device='cpu',n_jobs=self.cfg.threads,random_state=self.seed+7101,
            lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=8,verbosity=0)
        self.map_model.fit(np.column_stack([X[ix],family_input]),y[ix],group=groups)
        # V5's useful supervision separation: positive annotations versus confirmed
        # non-target hands ONLY. Unknown and unannotated positive-pair hands are excluded.
        witness=(y==1)|neg
        self.witness_indices=np.array([i for i,n in enumerate(EVIDENCE_FEATURES) if
            n.startswith(('null_','partner_','hu_','outsider_','voluntary_','query_'))
            or n.startswith(('river_','rule_'))],np.int32)
        self.witness=lgb.LGBMClassifier(**{**lgb_params(self.cfg,self.seed+7201,self.cfg.witness_trees),
            'num_leaves':11,'reg_lambda':16.,'min_child_samples':24})
        wy=y[witness]
        self.witness.fit(X[witness][:,self.witness_indices],wy,
            sample_weight=np.where(wy,8.,1.)*np.sqrt(qw[witness]))
        self.witness_reference=fit_score_reference(self.witness.booster_.predict(XC[:,self.witness_indices],num_threads=self.cfg.threads)) if len(XC) else (0.,1.)
        self.supervision_report=dict(fit_rows=int(train.sum()),reference_rows=int(reference.sum()),
            witness_positive=int(wy.sum()),witness_confirmed_non_target=int((wy==0).sum()),
            unannotated_positive_hands_excluded=int(((y==0)&~neg).sum()),
            unknown_hands_in_witness=0,rank_queries=len(groups),rank_family_input='hypothesis one-hot, not an identifier')
        return self
    def predict(self,X):return super().predict(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_likelihood(self,X):return super().predict_likelihood(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_generic(self,X):return super().predict_generic(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_additions(self,X,with_map=True):
        if not len(X):return np.empty((0,3),np.float32),np.empty((0,3),np.float32),np.empty(0,np.float32)
        q=np.column_stack([normalize_score(model.booster_.predict(X,num_threads=self.cfg.threads),ref)
                           for model,ref in zip(self.query_models,self.query_refs)]).astype(np.float32)
        ranks=np.zeros((len(X),3),np.float32)
        if with_map:
            for f in range(3):
                hypothesis=np.zeros((len(X),3),np.float32);hypothesis[:,f]=1.
                ranks[:,f]=self.map_model.predict(np.column_stack([X,hypothesis]))
        w=normalize_score(self.witness.booster_.predict(X[:,self.witness_indices],num_threads=self.cfg.threads),self.witness_reference)
        return q,ranks,w




--- CODE CELL 24 ---
LEARNED_STATS=['mean','max','q95','top3','top5','top10','above01','above03','above06','peak3','peak9','peak21','excess_top5']
LEARNED_FEATURES=[f'evidence_{f}_{stat}' for f in [*FAMILIES,'any'] for stat in LEARNED_STATS]
LEARNED_FEATURES += ['evidence_disclosed_entropy','evidence_disclosed_margin','evidence_episode_top5_span']


ANCHOR_LEARNED_FEATURES=list(LEARNED_FEATURES)
LEARNED_FEATURES += [n.replace('evidence_','query_evidence_',1) for n in ANCHOR_LEARNED_FEATURES]
LEARNED_FEATURES += ['witness_'+stat for stat in LEARNED_STATS]
ANCHOR_PAIR_WIDTH=len(PAIR_FEATURES)+len(ANCHOR_LEARNED_FEATURES)

def anchor_pair_features(c,progress):
    result=[];n=len(c)
    if n==0:return np.zeros(len(ANCHOR_LEARNED_FEATURES),np.float32)
    matrix=np.column_stack([c,c.max(axis=1)])
    for f in range(4):
        v=matrix[:,f];sort=np.sort(v)[::-1]
        vals=[v.mean(),sort[0],np.quantile(v,.95),sort[:3].mean(),sort[:5].mean(),sort[:10].mean(),
              np.mean(v>.1),np.mean(v>.3),np.mean(v>.6)]
        for width in (3,9,21):
            w=min(width,n);cs=np.r_[0,np.cumsum(v,dtype=np.float64)]
            vals.append(np.max((cs[w:]-cs[:-w])/w))
        vals.append(sort[:5].mean()-np.median(v));result.extend(vals)
    q=np.maximum(c.mean(axis=0),1e-7);q/=q.sum();qq=np.sort(q)
    order=np.argsort(-matrix[:,-1],kind='stable')[:5]
    result.extend([-np.sum(q*np.log(q)),qq[-1]-qq[-2],np.ptp(progress[order]) if len(order)>1 else 0.])
    return np.asarray(result,np.float32)


def evidence_pair_features(c,progress,q=None,w=None):
    old=anchor_pair_features(c,progress)
    if q is None or w is None:
        raise ValueError('V6 learned pair features require query and witness scores; no silent V4 fallback.')
    extra=anchor_pair_features(q,progress)
    ww=anchor_pair_features(np.repeat(np.asarray(w)[:,None],3,axis=1),progress)[:len(LEARNED_STATS)]
    return np.r_[old,extra,ww].astype(np.float32)



def chronological_view(x,fraction=2/3):
    if not len(x):return x.copy(),0
    start=max(0,len(x)-max(1,int(np.ceil(len(x)*fraction))))
    out=x[start:].copy();v=out[:,HF['progress']]
    if len(v)>1:out[:,HF['progress']]=(v-v[0])/max(1e-9,float(v[-1]-v[0]))
    else:out[:,HF['progress']]=0.
    return out,start


def matched_raw_features(xx,profile):
    part,start=chronological_view(xx)
    return aggregate_one(part,profile),start


def score_blocks(experts,paths,pairs,target_mask,root,cfg,expected_disjoint=True,tail_fraction=1.,need_augmentation=True):
    """All shared hands by default. Short-view audit rebuilds query features and predictions."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    n=len(pairs);L=np.zeros((n,len(LEARNED_FEATURES)),np.float32)
    do_aug=bool(need_augmentation and cfg.temporal_augmentation and tail_fraction==1.)
    A=np.zeros_like(L) if do_aug else np.empty((0,len(LEARNED_FEATURES)),np.float32)
    raw_aug=np.zeros((n,len(PAIR_FEATURES)),np.float32) if do_aug else np.empty((0,len(PAIR_FEATURES)),np.float32)
    files=[];view_raw=np.zeros((n,len(PAIR_FEATURES)),np.float32) if tail_fraction<1 else None
    for path in paths:
        phase=int(path.name.split('_')[1]);pool=int(path.stem.split('_')[-1])
        if not np.any(target_mask&(pairs.pool.to_numpy()==pool)&(pairs.phase.to_numpy()==phase)):continue
        if expected_disjoint:
            for e in experts:assert pool not in e.training_pools,'Upstream evidence-label leakage across pools'
        with np.load(path) as z:
            allids=z['pids'];rows=z['local_pid'];raw=z['x'];hs=z['hid'];bounds=group_ranges(rows,len(allids))
            mat=[];kept=[];newraw=[];newhs=[];counts=[];profiles=[];augmat=[];augcounts=[];augparts=[]
            for j,p in enumerate(allids):
                if not target_mask[p]:continue
                a,b=bounds[j:j+2];xx=raw[a:b];hh=hs[a:b]
                if tail_fraction<1:
                    xx,cut=chronological_view(xx,tail_fraction);hh=hh[cut:]
                mat.append(evidence_matrix(xx));kept.append(p);newraw.append(xx);newhs.append(hh);counts.append(len(xx))
                profile=contextual_profile_features(j,int(pairs.loc[p,'u_local']),int(pairs.loc[p,'v_local']),z['total_prof'],z['shared_prof']) if do_aug or tail_fraction<1 else None
                profiles.append(profile)
                short,_=chronological_view(xx) if do_aug else (np.empty((0,xx.shape[1]),np.float32),0)
                augparts.append(short);augcounts.append(len(short))
                if do_aug:augmat.append(evidence_matrix(short))
            if not kept:continue
            X=np.concatenate(mat);xx=np.concatenate(newraw);hh=np.concatenate(newhs);ids=np.array(kept,np.int32)
            starts=np.r_[0,np.cumsum(counts)];astarts=np.r_[0,np.cumsum(augcounts)]
            c=np.zeros((len(X),3),np.float32);r=c.copy();m=c.copy();g=np.zeros(len(X),np.float32)
            ac=np.zeros((int(astarts[-1]),3),np.float32)
            q=c.copy();t=c.copy();w=np.zeros(len(X),np.float32)
            aq=ac.copy();aw=np.zeros(int(astarts[-1]),np.float32)
            AX=np.concatenate(augmat) if augmat else None
            for expert in experts:
                cp,rp,mp=expert.predict(X);c+=cp/len(experts);g+=expert.predict_generic(X)/len(experts)
                qp,tp,wp=expert.predict_additions(X);q+=qp/len(experts);w+=wp/len(experts)
                if AX is not None:
                    ac+=expert.predict_likelihood(AX)/len(experts)
                    aqp,_,awp=expert.predict_additions(AX,with_map=False);aq+=aqp/len(experts);aw+=awp/len(experts)
                for j in range(len(ids)):
                    a,b=starts[j:j+2]
                    for f in range(3):
                        r[a:b,f]+=percentiles(rp[a:b,f])/len(experts)
                        m[a:b,f]+=percentiles(mp[a:b,f])/len(experts)
                        t[a:b,f]+=percentiles(tp[a:b,f])/len(experts)
            d=(1-np.exp(-np.maximum(0,xx[:,HF['null_action_score']])/4.)).astype(np.float32)
            rule=np.zeros((len(X),4),np.float32)
            for j,p in enumerate(ids):
                a,b=starts[j:j+2];part=xx[a:b];prog=part[:,HF['progress']]
                L[p]=evidence_pair_features(c[a:b],prog,q[a:b],w[a:b])
                if do_aug:raw_aug[p]=aggregate_one(augparts[j],profiles[j])
                if AX is not None:
                    ca,cb=astarts[j:j+2];A[p]=evidence_pair_features(ac[ca:cb],augparts[j][:,HF['progress']],aq[ca:cb],aw[ca:cb])
                elif do_aug:
                    cut=len(part)-len(augparts[j]);A[p]=evidence_pair_features(c[a+cut:b],augparts[j][:,HF['progress']],q[a+cut:b],w[a+cut:b])
                if view_raw is not None:view_raw[p]=aggregate_one(part,profiles[j])
                for f in range(4):rule[a:b,f]=percentiles(part[:,RULE_IDX[f]])
            target=root/path.name
            np.savez_compressed(target,pids=ids,counts=np.array(counts,np.int32),hid=hh,c=c,r=r,m=m,rule=rule,g=g,d=d,q=q,t=t,w=w,progress=xx[:,HF['progress']])
            files.append(target)
        if len(files)%25==0:log('full_hand_scan',stage_folder=root.name,blocks=len(files),models=len(experts),tail_fraction=tail_fraction)
    result=dict(learned=L,aug_learned=A,aug_raw=raw_aug,files=files)
    if view_raw is not None:result['view_raw']=view_raw
    return result






def read_records(files,pairs,gold,positives_only=True):
    rec=[]
    for path in files:
        with np.load(path) as z:
            starts=np.r_[0,np.cumsum(z['counts'])]
            for j,p in enumerate(z['pids']):
                if positives_only and pairs.loc[p,'label']!=1:continue
                a,b=starts[j:j+2]
                rec.append(dict(pid=int(p),hid=z['hid'][a:b],c=z['c'][a:b],r=z['r'][a:b],m=z['m'][a:b],
                                rule=z['rule'][a:b],g=z['g'][a:b],d=z['d'][a:b],q=z['q'][a:b],t=z['t'][a:b],w=z['w'][a:b],gold=gold.get(int(p),set()),progress=z['progress'][a:b]))
    return rec




--- CODE CELL 26 ---
@njit(cache=False)
def episode_posterior(values):
    """Fixed two-state retrospective smoother. Descriptive support, not fitted latent truth."""
    n=len(values);out=np.zeros(n,np.float64)
    if n==0:return out
    forward=np.zeros((n,2),np.float64);emit=np.zeros((n,2),np.float64)
    # inactive->active .025; active->inactive .20. Never tuned on an outer pool.
    t00=.975;t01=.025;t10=.20;t11=.80
    for i in range(n):
        p=min(.9999,max(.0001,float(values[i])))
        ll=max(-5.,min(5.,np.log(p/(1-p))))
        emit[i,0]=1.;emit[i,1]=np.exp(ll)
        if i==0:q0=.92;q1=.08
        else:q0=forward[i-1,0]*t00+forward[i-1,1]*t10;q1=forward[i-1,0]*t01+forward[i-1,1]*t11
        a=q0;b=q1*emit[i,1];den=max(1e-30,a+b)
        forward[i,0]=a/den;forward[i,1]=b/den
    b0=1.;b1=1.
    for i in range(n-1,-1,-1):
        den=max(1e-30,forward[i,0]*b0+forward[i,1]*b1)
        out[i]=forward[i,1]*b1/den
        if i>0:
            c0=t00*b0+t01*emit[i,1]*b1;c1=t10*b0+t11*emit[i,1]*b1
            den=max(1e-30,c0+c1);b0=c0/den;b1=c1/den
    return out


EXPERT_NAMES=['likelihood_rank','ndcg_rank','ap5_rank','likelihood_margin','episode_gated','gameplay_rule','generic_evidence','action_null']

def evidence_experts(c,r,rule,f,ap=None,generic=None,null=None):
    v=c[:,f];cls=percentiles(v);rank=r[:,f];amap=rank if ap is None else ap[:,f]
    lo=float(v.min()) if len(v) else 0.;hi=float(v.max()) if len(v) else 0.
    margin=(v-lo)/max(1e-6,hi-lo)
    # The observed hand score remains multiplicative: temporal activation alone is never evidence.
    episode=percentiles(v*(.70+.30*episode_posterior(v)))
    g=c.max(axis=1) if generic is None else generic
    d=rule[:,3] if null is None else null
    return np.column_stack([cls,rank,amap,margin,episode,rule[:,f],percentiles(g),percentiles(d)])

class AnchorEvidencePolicy:
    def fit(self,records,pairs):
        base=np.array([.25,.75,0,0,0,0,0,0.],float)
        self.weights=np.tile(base,(3,1));self.power=1.;self.report=[]
        candidates=[base]
        for j in (0,1,2,3,4,6):
            w=np.zeros(8);w[j]=1.;candidates.append(w)
            w=.75*base.copy();w[j]+=.25;candidates.append(w)
        for j in (5,7):
            w=.90*base.copy();w[j]+=.10;candidates.append(w)
        for f in range(3):
            rr=[q for q in records if pairs.loc[q['pid'],'family']==FAMILIES[f]]
            if not rr:continue
            ex=[evidence_experts(q['c'],q['r'],q['rule'],f,q['m'],q.get('g'),q.get('d')) for q in rr]
            rew=np.array([[ap5(q['hid'][np.argsort(-(e@w),kind='stable')[:5]].tolist(),q['gold']) for q,e in zip(rr,ex)] for w in candidates])
            means=rew.mean(axis=1);best=int(np.argmax(means));pools=np.array([pairs.loc[q['pid'],'pool'] for q in rr]);uniq=np.unique(pools)
            delta=np.array([(rew[best,pools==t]-rew[0,pools==t]).mean() for t in uniq])
            se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 1.
            chosen=best if means[best]>means[0]+se+.001 else 0
            self.weights[f]=candidates[chosen]
            self.report.append(dict(family=str(FAMILIES[f]),queries=len(rr),pools=len(uniq),selected_weights=self.weights[f].tolist(),
                selected_inner_map5=float(means[chosen]),retained_inner_map5=float(means[0]),paired_pool_gain_se=se,candidate_count=len(candidates),
                expert_map5={EXPERT_NAMES[j]:float(np.mean([ap5(q['hid'][np.argsort(-e[:,j],kind='stable')[:5]].tolist(),q['gold']) for q,e in zip(rr,ex)])) for j in range(8)}))
        routing=[q for q in records if 'routing' in q];self.routing_report=[]
        for power in (1.,2.,4.):
            vals=[]
            for q in routing:
                fp=np.maximum(q['routing'],1e-9)**power;fp/=fp.sum()
                matrix=np.column_stack([evidence_experts(q['c'],q['r'],q['rule'],f,q['m'],q.get('g'),q.get('d'))@self.weights[f] for f in range(3)])
                vals.append(ap5(q['hid'][np.argsort(-(matrix@fp),kind='stable')[:5]].tolist(),q['gold']))
            if vals:self.routing_report.append(dict(probability_power=power,inner_routing_map5=float(np.mean(vals))))
        if self.routing_report:
            best=max(self.routing_report,key=lambda z:z['inner_routing_map5'])
            if best['inner_routing_map5']>self.routing_report[0]['inner_routing_map5']+.002:self.power=best['probability_power']
        return self
    def rank(self,record,fp,other=False):
        if not len(record['hid']):return []
        fp=np.maximum(fp,1e-9)**self.power;fp/=fp.sum();score=np.zeros(len(record['hid']),float)
        for f in range(3):score+=fp[f]*(evidence_experts(record['c'],record['r'],record['rule'],f,record.get('m'),record.get('g'),record.get('d'))@self.weights[f])
        if other:
            g=record.get('g',record['c'].max(axis=1));d=record.get('d',record['rule'][:,3])
            score=.25*score+.45*percentiles(g)+.30*percentiles(d)
        return record['hid'][np.argsort(-score,kind='stable')[:5]].astype(int).tolist()



def paired_pool_gate(delta,pools,min_queries=12,min_pools=6,seed=6181,quantile=.025):
    """Query-weighted paired bootstrap over entire pools; a gate, not a performance guarantee."""
    delta=np.asarray(delta,float);pools=np.asarray(pools)
    unique=np.unique(pools)
    if len(delta)<min_queries or len(unique)<min_pools:
        return dict(passed=False,mean_gain=float(delta.mean()) if len(delta) else 0.,lower=None,
                    queries=len(delta),pools=len(unique),reason='insufficient independent validation support')
    sums=np.array([delta[pools==p].sum() for p in unique]);counts=np.array([(pools==p).sum() for p in unique])
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(unique),(800,len(unique)))
    draws=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    low=float(np.quantile(draws,quantile))
    return dict(passed=bool(delta.mean()>.001 and low>0),mean_gain=float(delta.mean()),lower=low,
                queries=len(delta),pools=len(unique),bootstrap_draws=800,lower_quantile=quantile,
                reason='paired pool-held-out test')

NEW_EXPERT_NAMES=['query_classifier','pooled_map','behavioral_witness','query_map_consensus']

def challenger_evidence_experts(record,f):
    for k in ('q','t','w'):
        if k not in record:raise ValueError('Missing V6 score channel: '+k)
    q=percentiles(record['q'][:,f]);t=record['t'][:,f];w=percentiles(record['w'])
    return np.column_stack([q,t,w,.5*q+.5*t])

class EvidencePolicy(AnchorEvidencePolicy):
    """Pick proposals on inner search pools; admit on a separate inner gate fold."""
    def fit(self,records,pairs):
        gate_fold=CFG.n_folds-1
        # Full-inner V4 policy is retained as a control, not advertised as the historical CSV.
        # Proposals still use only search folds; control comparison is an additional,
        # conservative diagnostic (the control selector itself has seen the gate labels).
        self.control=AnchorEvidencePolicy().fit(records,pairs)
        search=[r for r in records if int(pairs.loc[r['pid'],'fold'])!=gate_fold]
        gate=[r for r in records if int(pairs.loc[r['pid'],'fold'])==gate_fold]
        if not search:raise ValueError('No evidence-policy search queries')
        super().fit(search,pairs)
        self.challenger=np.zeros((3,2),float);self.gate_report=[]
        for f in range(3):
            train=[r for r in search if pairs.loc[r['pid'],'family']==FAMILIES[f]]
            valid=[r for r in gate if pairs.loc[r['pid'],'family']==FAMILIES[f]]
            if not train or not valid:continue
            old=[self.anchor_scores(r,f) for r in train]
            extras=[challenger_evidence_experts(r,f) for r in train]
            baseline=np.array([ap5(r['hid'][np.argsort(-v,kind='stable')[:5]].tolist(),r['gold']) for r,v in zip(train,old)])
            proposals=[]
            for expert in range(4):
                for alpha in (.25,.50,.75):
                    reward=np.array([ap5(r['hid'][np.argsort(-((1-alpha)*v+alpha*e[:,expert]),kind='stable')[:5]].tolist(),r['gold']) for r,v,e in zip(train,old,extras)])
                    proposals.append((float(reward.mean()),expert,alpha))
            best=max(proposals,key=lambda z:z[0]);expert,alpha=best[1:]
            before=[];after=[];control_before=[]
            for r in valid:
                v=self.anchor_scores(r,f);e=challenger_evidence_experts(r,f)
                control_v=self.anchor_scores(r,f,control=True)
                control_before.append(ap5(r['hid'][np.argsort(-control_v,kind='stable')[:5]].tolist(),r['gold']))
                before.append(ap5(r['hid'][np.argsort(-v,kind='stable')[:5]].tolist(),r['gold']))
                after.append(ap5(r['hid'][np.argsort(-((1-alpha)*v+alpha*e[:,expert]),kind='stable')[:5]].tolist(),r['gold']))
            support=paired_pool_gate(np.array(after)-before,[pairs.loc[r['pid'],'pool'] for r in valid],
                                    CFG.gate_min_queries,CFG.gate_min_pools,seed=CFG.seed+f)
            admitted=bool(best[0]>baseline.mean()+.002 and support['passed'] and np.mean(after)>np.mean(control_before)+.001)
            if admitted:self.challenger[f]=(expert,alpha)
            self.gate_report.append(dict(family=str(FAMILIES[f]),proposal=NEW_EXPERT_NAMES[expert],proposed_weight=alpha,
                selected_weight=alpha if admitted else 0.,search_queries=len(train),search_anchor_map5=float(baseline.mean()),
                search_proposal_map5=best[0],gate_anchor_map5=float(np.mean(before)),gate_proposal_map5=float(np.mean(after)),
                gate=support,gate_v4_control_map5=float(np.mean(control_before)),admitted=admitted))
        for f in range(3):
            if self.challenger[f,1]==0:self.weights[f]=self.control.weights[f]
        self.power=self.control.power
        return self
    def anchor_scores(self,record,f,control=False):
        return evidence_experts(record['c'],record['r'],record['rule'],f,record.get('m'),record.get('g'),record.get('d'))@(self.control.weights[f] if control else self.weights[f])
    def rank(self,record,fp,other=False,anchor_only=False):
        if not len(record['hid']):return []
        fp=np.maximum(fp,1e-9)**self.power;fp/=fp.sum();score=np.zeros(len(record['hid']),float)
        for f in range(3):
            v=self.anchor_scores(record,f,control=anchor_only);expert,alpha=self.challenger[f]
            if not anchor_only and alpha>0:
                v=(1-alpha)*v+alpha*challenger_evidence_experts(record,f)[:,int(expert)]
            score+=fp[f]*v
        if other:
            g=record.get('g',record['c'].max(axis=1));d=record.get('d',record['rule'][:,3])
            score=.25*score+.45*percentiles(g)+.30*percentiles(d)
        return record['hid'][np.argsort(-score,kind='stable')[:5]].astype(int).tolist()





--- CODE CELL 28 ---
def raw_routing_oof(rawX,pairs,mask,folds,cfg):
    """No upstream learned-evidence features here, avoiding transitive label leakage during policy tuning."""
    probabilities={}
    for fold in np.unique(folds[mask]):
        tr=mask&(folds!=fold)&(pairs.label.to_numpy()==1)
        va=mask&(folds==fold)&(pairs.label.to_numpy()==1)
        train=np.flatnonzero(tr);valid=np.flatnonzero(va)
        if not len(valid):continue
        y=np.array([int(np.flatnonzero(FAMILIES==f)[0]) for f in pairs.loc[train,'family']])
        if len(np.unique(y))<3:continue
        model=lgb.LGBMClassifier(**{**lgb_params(cfg,cfg.seed+901+int(fold),max(60,cfg.risk_trees//2)),
                                  'num_leaves':9,'min_child_samples':10})
        model.fit(rawX[train],y)
        pp=model.booster_.predict(rawX[valid],num_threads=cfg.threads)
        probabilities.update({int(pid):p for pid,p in zip(valid,pp)})
    return probabilities

def leave_one_family_out_audit(rawX,pairs,train_mask,valid_mask,cfg):
    """Diagnostic only: simulate unseen-family classification using raw gameplay. Never tunes the submitted model."""
    results=[];yv=pairs.label.to_numpy();fv=pairs.family.to_numpy()
    for f in FAMILIES:
        tr=train_mask&(yv>=0)&((yv==0)|(fv!=f));va=valid_mask&((yv==0)|(fv==f))
        ti=np.flatnonzero(tr);vi=np.flatnonzero(va)
        if len(vi)==0 or len(np.unique(yv[vi]))<2:continue
        m=lgb.LGBMClassifier(**{**lgb_params(cfg,cfg.seed+577,int(max(50,cfg.risk_trees//2))),'num_leaves':11})
        m.fit(rawX[ti],yv[ti]);score=m.booster_.predict(rawX[vi],num_threads=cfg.threads)
        results.append(dict(withheld_family=str(f),training_positive_family_excluded=True,
            known_negative_and_withheld_positive_ap=ranked_ap(yv[vi],score,pairs.loc[vi,'pair_id']),
            evaluation_pairs=len(vi),withheld_positives=int(yv[vi].sum())))
    return {'results':results,'status':'OUTER_POOL_DIAGNOSTIC_ONLY_NOT_A_MODEL_SELECTION_SCORE',
            'warning':'Disclosed-family withholding is not validation of the undisclosed mechanism.'}

class AnchorPairModel:
    def __init__(self,cfg,seed):self.cfg=cfg;self.seed=int(seed)
    def fit(self,X,y,family,exposure_weight=None):
        known=y>=0;pos=y==1;par=lgb_params(self.cfg,self.seed,self.cfg.risk_trees)
        if min(pos.sum(),(y==0).sum())<3:raise ValueError('Insufficient public positive/negative labels')
        mult=np.ones(len(y)) if exposure_weight is None else np.asarray(exposure_weight)
        self.pn=lgb.LGBMClassifier(**par)
        self.pn.fit(X[known],y[known],sample_weight=np.where(y[known]==1,3.,1.)*mult[known])
        self.cat=None
        if self.cfg.pair_catboost:
            try:from catboost import CatBoostClassifier
            except ImportError:log('optional_pair_catboost_unavailable',fallback='LightGBM only; no silent installation')
            else:
                self.cat=CatBoostClassifier(iterations=self.cfg.pair_catboost_trees,depth=5,learning_rate=.04,
                    l2_leaf_reg=10.,loss_function='Logloss',thread_count=self.cfg.threads,random_seed=self.seed,
                    verbose=False,allow_writing_files=False,task_type='CPU')
                self.cat.fit(X[known],y[known],sample_weight=np.where(y[known]==1,3.,1.)*mult[known])
        self.pu=lgb.LGBMClassifier(**{**par,'random_state':self.seed+71,'num_leaves':23,'min_child_samples':32})
        w=np.where(pos,3.,np.where(y==0,1.,self.cfg.pu_unknown_weight))*mult
        unknown=y<0
        if unknown.any():w[unknown]*=(.25+.75*(1.-self.pn.booster_.predict(X[unknown],num_threads=self.cfg.threads)))
        self.pu.fit(X,pos.astype(np.int8),sample_weight=w)
        fy=np.array([int(np.flatnonzero(FAMILIES==f)[0]) for f in family[pos]],np.int32)
        self.family_classes=np.unique(fy)
        # Two classes are allowed only for the explicitly marked withheld-family diagnostic.
        if len(self.family_classes)<2:raise ValueError('At least two known target families are required')
        yy=np.searchsorted(self.family_classes,fy);cnt=np.bincount(yy)
        self.family=lgb.LGBMClassifier(**{**par,'num_leaves':9,'n_estimators':max(80,int(self.cfg.risk_trees*.65)),'min_child_samples':12})
        self.family.fit(X[pos],yy,sample_weight=mult[pos]*len(fy)/(len(cnt)*cnt[yy]))
        self.mechanism_indices=np.array([i for i,n in enumerate(PAIR_FEATURES) if n.startswith(('null_','ctx_'))
            or n in ('log_shared','flow_direction_imbalance','surrender_action_coherence','transfer_action_coherence')],np.int32)
        self.mechanism=lgb.LGBMClassifier(**{**par,'num_leaves':9,'min_child_samples':24,'n_estimators':self.cfg.mechanism_trees})
        self.mechanism.fit(X[known][:,self.mechanism_indices],y[known],sample_weight=np.where(y[known]==1,2.,1.)*mult[known])
        self.null_score_index=PAIR_FEATURES.index('null_action_score_top5')
        self.null_count_index=PAIR_FEATURES.index('null_rare_benefit_count_mean')
        self.exposure_index=PAIR_FEATURES.index('log_shared')
        self.null_cut=max(.5,float(np.quantile(X[y==0,self.null_score_index],.995)))
        self.blend=tuple(self.cfg.pair_blend);self.open_weight=float(self.cfg.open_set_weight)
        return self
    def predict(self,X):
        pn=self.pn.booster_.predict(X,num_threads=self.cfg.threads)
        pu=self.pu.booster_.predict(X,num_threads=self.cfg.threads)
        cat=pn if self.cat is None else self.cat.predict_proba(X,thread_count=self.cfg.threads)[:,1]
        fp0=self.family.booster_.predict(X,num_threads=self.cfg.threads)
        if np.ndim(fp0)==1:fp0=np.column_stack([1-fp0,fp0])
        fp=np.zeros((len(X),3),float)
        fp[:,self.family_classes]=fp0
        mech=self.mechanism.booster_.predict(X[:,self.mechanism_indices],num_threads=self.cfg.threads)
        base=self.blend[0]*pn+self.blend[1]*pu+self.blend[2]*cat
        # At least two observable low-reference-probability partner-benefiting decisions.
        supporting=X[:,self.null_count_index]*np.expm1(np.clip(X[:,self.exposure_index],0,20))
        gate=(X[:,self.null_score_index]>self.null_cut)&(supporting>=2.-1e-4)&(mech>.40)
        promotion=self.open_weight*(1-base)*mech*gate
        risk=np.clip(base+promotion,1e-12,1-1e-12)
        other=(promotion>np.maximum(.015,.50*base))&gate&self.cfg.other_enabled
        return dict(risk=risk,family=fp,other=other,pn=pn,pu=pu,cat=cat,
                    mechanism=mech,open_gate=gate,risk_base=base)


class PairModel(AnchorPairModel):
    def fit(self,X,y,family,exposure_weight=None):
        self.anchor_width=min(X.shape[1],ANCHOR_PAIR_WIDTH)
        super().fit(X[:,:self.anchor_width],y,family,exposure_weight)
        self.challenger_pn=None;self.challenger_pu=None
        self.v6_weight=float(self.cfg.v6_pair_weight)
        if X.shape[1]>ANCHOR_PAIR_WIDTH:
            known=y>=0;pos=y==1;mult=np.ones(len(y)) if exposure_weight is None else np.asarray(exposure_weight)
            par={**lgb_params(self.cfg,self.seed+8151,self.cfg.challenger_pair_trees),
                 'num_leaves':11,'min_child_samples':30,'reg_lambda':18.}
            self.challenger_pn=lgb.LGBMClassifier(**par)
            self.challenger_pn.fit(X[known],y[known],sample_weight=np.where(pos[known],3.,1.)*mult[known])
            weight=np.where(pos,3.,np.where(y==0,1.,self.cfg.pu_unknown_weight))*mult
            unknown=y<0
            if unknown.any():weight[unknown]*=.25+.75*(1-self.challenger_pn.booster_.predict(X[unknown],num_threads=self.cfg.threads))
            self.challenger_pu=lgb.LGBMClassifier(**{**par,'random_state':self.seed+8251,'num_leaves':19})
            self.challenger_pu.fit(X,pos.astype(np.int8),sample_weight=weight)
        return self
    def predict(self,X):
        p=super().predict(X[:,:self.anchor_width]);anchor=p['risk'].copy()
        candidate=anchor.copy()
        if self.challenger_pn is not None:
            pn=self.challenger_pn.booster_.predict(X,num_threads=self.cfg.threads)
            pu=self.challenger_pu.booster_.predict(X,num_threads=self.cfg.threads)
            candidate=.2*pn+.8*pu
        p['risk_anchor']=anchor;p['risk_challenger']=candidate
        p['risk']=np.clip((1-self.v6_weight)*anchor+self.v6_weight*candidate,1e-12,1-1e-12)
        return p



def policy_weighted_ap(y,risk,ids,prior):
    y=np.asarray(y);keep=y>=0;y=y[keep];risk=np.asarray(risk)[keep];ids=np.asarray(ids)[keep]
    if not np.any(y==1) or not np.any(y==0):return 0.
    order=np.lexsort((ids.astype(str),-risk));yy=y[order]
    w=np.where(yy==1,prior/(y==1).sum(),(1-prior)/(y==0).sum())
    tp=np.cumsum(w*yy);pr=tp/np.cumsum(w)
    return float(np.sum(pr*w*yy)/max(1e-12,tp[-1]))


def select_pair_policy(rawX,pairs,tune,folds,cfg):
    """All selection is RAW-only and excludes outer pool 0; no upstream-stacking label leakage."""
    from dataclasses import replace
    cc=replace(cfg,risk_trees=min(260,cfg.risk_trees),mechanism_trees=min(180,cfg.mechanism_trees),
               pair_catboost_trees=min(220,cfg.pair_catboost_trees),open_set_weight=0.)
    ids=np.flatnonzero(tune);preds={};pools=pairs.pool.to_numpy();yy=pairs.label.to_numpy();fam=pairs.family.to_numpy()
    for fold in np.unique(folds[tune]):
        tr=tune&(folds!=fold);va=tune&(folds==fold);ti=np.flatnonzero(tr);vi=np.flatnonzero(va)
        model=PairModel(cc,cfg.seed+3067+int(fold)).fit(rawX[ti],yy[ti],fam[ti])
        q=model.predict(rawX[vi])
        for j,p in enumerate(vi):preds[int(p)]={k:(v[j].copy() if isinstance(v[j],np.ndarray) else v[j]) for k,v in q.items()}
    pn=np.array([preds[int(p)]['pn'] for p in ids]);pu=np.array([preds[int(p)]['pu'] for p in ids]);cat=np.array([preds[int(p)]['cat'] for p in ids])
    candidates=[(.15,.70,.15),(.30,.50,.20),(.10,.80,.10),(.20,.60,.20)]
    rows=[];scores=[];risks=[]
    for w in candidates:
        r=w[0]*pn+w[1]*pu+w[2]*cat;known=yy[ids]>=0
        pap=ranked_ap(yy[ids][known],r[known],pairs.loc[ids[known],'pair_id'])
        a005=policy_weighted_ap(yy[ids],r,pairs.loc[ids,'pair_id'],.005)
        a02=policy_weighted_ap(yy[ids],r,pairs.loc[ids,'pair_id'],.02)
        score=.50*pap+.25*a005+.25*a02
        scores.append(score);risks.append(r);rows.append(dict(weights=w,pair_ap=pap,prior_0005_ap=a005,prior_002_ap=a02,selection_utility=score))
    best=int(np.argmax(scores));chosen=best if scores[best]>scores[0]+.001 else 0
    chosen_blend=candidates[chosen];base=risks[chosen]
    # Withhold a family on a distinct inner validation fold. All inputs remain label-free raw gameplay.
    held=int(np.unique(folds[tune])[-1]);results=[];gain_by_alpha={.15:[],.30:[]}
    for missing in FAMILIES:
        tr=tune&(folds!=held)&((yy!=1)|(fam!=missing))
        va=tune&(folds==held)&((yy==0)|(fam==missing));ti=np.flatnonzero(tr);vi=np.flatnonzero(va)
        if not len(vi) or len(np.unique(yy[vi]))<2:continue
        mc=replace(cc,pair_blend=chosen_blend,pair_catboost=False)
        model=PairModel(mc,cfg.seed+4100+int(np.flatnonzero(FAMILIES==missing)[0])).fit(rawX[ti],yy[ti],fam[ti])
        q=model.predict(rawX[vi]);r=q['risk_base'];ap0=ranked_ap(yy[vi],r,pairs.loc[vi,'pair_id'])
        row=dict(withheld_family=str(missing),baseline_ap=ap0,positives=int((yy[vi]==1).sum()),gated_pairs=int(q['open_gate'].sum()),promoted={})
        for alpha in (.15,.30):
            score=r+alpha*(1-r)*q['mechanism']*q['open_gate']
            val=ranked_ap(yy[vi],score,pairs.loc[vi,'pair_id']);gain_by_alpha[alpha].append(val-ap0);row['promoted'][str(alpha)]=val
        results.append(row)
    selected_alpha=0.;alpha_report=[]
    for alpha,gains in gain_by_alpha.items():
        mech=np.array([preds[int(p)]['mechanism'] for p in ids]);gate=np.array([preds[int(p)]['open_gate'] for p in ids])
        promoted=base+alpha*(1-base)*mech*gate
        normal_utility=.50*ranked_ap(yy[ids][yy[ids]>=0],promoted[yy[ids]>=0],pairs.loc[ids[yy[ids]>=0],'pair_id'])+.25*policy_weighted_ap(yy[ids],promoted,pairs.loc[ids,'pair_id'],.005)+.25*policy_weighted_ap(yy[ids],promoted,pairs.loc[ids,'pair_id'],.02)
        median=float(np.median(gains)) if gains else 0.;minimum=float(np.min(gains)) if gains else -1.
        passed=len(gains)==3 and median>.005 and minimum>=-.005 and normal_utility>=scores[chosen]-.002
        alpha_report.append(dict(alpha=alpha,withheld_median_gain=median,withheld_min_gain=minimum,
                                 known_family_utility=normal_utility,passes_gate=passed))
        if passed and selected_alpha==0:selected_alpha=alpha
    cfg.pair_blend=chosen_blend;cfg.open_set_weight=selected_alpha
    report=dict(status='INNER_RAW_ONLY_POLICY_SELECTION; outer fold 0 never read',pair_candidates=rows,
        selected_pair_blend=chosen_blend,withheld_family=results,open_set_candidates=alpha_report,
        selected_open_set_weight=selected_alpha,
        caveats=['Prevalence values are sensitivity assumptions, not prevalence estimates.',
                 'Withheld disclosed families do not validate the private mechanism.',
                 'Unknown pairs are never confirmed negatives in the selection metric.'])
    save_json(OUT/'frozen_pair_policy.json',report)
    log('pair_policy_frozen',weights=chosen_blend,open_set_weight=selected_alpha)
    return report




--- CODE CELL 30 ---
def augmented_training(pairs,ids,rawX,stage,gold):
    train_ids_set=set(map(int,ids))
    full=np.column_stack([rawX[ids],stage['learned'][ids]])
    y=pairs.loc[ids,'label'].to_numpy();fam=pairs.loc[ids,'family'].to_numpy();out_ids=list(ids)
    if not CFG.temporal_augmentation:return full,y,fam,np.ones(len(y)),np.array(out_ids)
    eligible=[]
    # Labels of potentially inactive cropped positives are never recycled blindly.
    for path in stage['files']:
        with np.load(path) as z:
            starts=np.r_[0,np.cumsum(z['counts'])]
            for j,p in enumerate(z['pids']):
                if int(p) not in train_ids_set:continue
                a,b=starts[j:j+2];cut=(b-a)//3
                if pairs.loc[p,'label']!=1 or set(z['hid'][a+cut:b].tolist())&gold.get(int(p),set()):eligible.append(int(p))
    eligible=np.asarray(sorted(set(eligible)),np.int32)
    if not len(eligible):return full,y,fam,np.ones(len(y)),np.array(out_ids)
    aug=np.column_stack([stage['aug_raw'][eligible],stage['aug_learned'][eligible]])
    weight=np.ones(len(ids));loc={int(p):i for i,p in enumerate(ids)}
    for p in eligible:weight[loc[int(p)]]=.5
    return (np.vstack([full,aug]),np.r_[y,pairs.loc[eligible,'label'].to_numpy()],
            np.r_[fam,pairs.loc[eligible,'family'].to_numpy()],np.r_[weight,np.full(len(eligible),.5)],np.r_[ids,eligible])


def pool_bootstrap_interval(pairs,ids,risk,behavior,evidence,gold,repeats=200,seed=173):
    # Resample whole pools rather than dependent pairs.
    ids=np.asarray(ids);known=ids[pairs.loc[ids,'label'].to_numpy()>=0]
    pools=np.unique(pairs.loc[known,'pool']);rng=np.random.default_rng(seed);vals=[]
    mapping={int(p):i for i,p in enumerate(ids)}
    for _ in range(repeats):
        select=np.concatenate([known[pairs.loc[known,'pool'].to_numpy()==t] for t in rng.choice(pools,len(pools),replace=True)])
        pframe=pairs.loc[select].copy();ix=np.array([mapping[int(p)] for p in select])
        r=metric_proxy(pframe,risk[ix],behavior[ix],evidence,gold)
        vals.append([r['pair_ap'],r['evidence_map5'],r['behavior_map'],r['composite_proxy']])
    q=np.quantile(vals,[.025,.975],axis=0)
    return {name:[float(q[0,i]),float(q[1,i])] for i,name in enumerate(['pair_ap','evidence_map5','behavior_map','composite_proxy'])}


def prevalence_stress(y,risk,ids):
    known=y>=0;yy=y[known];ss=risk[known];ii=ids[known]
    if min((yy==1).sum(),(yy==0).sum())==0:return {}
    order=np.lexsort((ii,-ss));yy=yy[order];out={}
    for prior in (.005,.01,.02,.05,.20):
        w=np.where(yy==1,prior/(yy==1).sum(),(1-prior)/(yy==0).sum())
        recall=np.cumsum(w*yy)/np.sum(w*yy);precision=np.cumsum(w*yy)/np.cumsum(w)
        out[str(prior)]=float(np.sum(np.diff(np.r_[0,recall])*precision))
    return out



--- CODE CELL 32 ---
def hand_packet(pack,pairs,pid,h):
    u=int(pairs.loc[pid,'u']);v=int(pairs.loc[pid,'v']);hv=pack.hand_values[h];bb=float(hv[0])
    seats=[]
    for seat in range(6):
        p=int(pack.seat_players[h,seat]);s=pack.seat_values[h,seat]
        role='A' if p==u else 'B' if p==v else f'O{seat}'
        seats.append(dict(role=role,seat=seat,cards=[CARD_INV[int(c)] for c in pack.seat_cards[h,seat]],
                          starting_stack_bb=float(s[0]/bb),contribution_bb=float(s[1]/bb),net_bb=float(s[2]/bb),folded=bool(s[4])))
    roles={int(pack.seat_players[h,seat]):seats[seat]['role'] for seat in range(6)}
    acts=[]
    for a in pack.actions[pack.action_ptr[h]:pack.action_ptr[h+1]]:
        st=int(a['street']);nb=0 if st==0 else st+2
        acts.append(dict(action_no=int(a['ano']),street=STREET_INV[st],role=roles[int(a['player'])],
                         action=ACTION_INV[int(a['kind'])],amount_bb=float(a['amount']/bb),
                         stack_before_bb=float(a['stack']/bb),to_call_bb=float(a['call']/bb),pot_before_bb=float(a['pot']/bb),
                         players_active=int(a['active']),visible_board=[CARD_INV[int(c)] for c in pack.board[h,:nb]]))
    return dict(hand_id=str(pack.hand_ids[h]),big_blind=bb,pot_bb=float(hv[3]/bb),seats=seats,actions=acts)

def observed_case_events(packet):
    """Literal, decision-time events only; no causal or private-label assertions."""
    live={r['role'] for r in packet['seats']};last=None;street=None;events=[]
    for a in packet['actions']:
        if a['street']!=street:street=a['street'];last=None
        role=a['role'];kind=a['action'];both={'A','B'}.issubset(live)
        aggressive=kind in ('bet','raise') or (kind=='all_in' and a['amount_bb']>a['to_call_bb'])
        context=f"{packet['hand_id']}, action {a['action_no']} ({street})"
        if both and role in ('A','B') and last in ('A','B') and last!=role and a['to_call_bb']>0:
            events.append(f"{context}: {role} chose {kind} facing {last}; the amount to call was {a['to_call_bb']:.2f} BB into a {a['pot_before_bb']:.2f} BB pot.")
        elif both and role not in ('A','B') and last in ('A','B') and kind=='fold' and a['to_call_bb']>0:
            events.append(f"{context}: outsider {role} folded facing {last}'s aggression while A and B were both still live.")
        elif both and role in ('A','B') and a['players_active']==2 and kind=='check':
            events.append(f"{context}: {role} checked with the partner as the only other live player; stack eligibility must also be reviewed.")
        if aggressive:last=role
        if kind=='fold':live.discard(role)
    return events

def make_case_reviews(pack,pairs,ev,risk,behavior,evid,cfg):
    order=np.argsort(-risk,kind='stable');chosen=[]
    for fam in [*FAMILIES,'other_coordination']:
        options=[i for i in order if behavior[i]==fam and evid.get(int(ev[i]))]
        if options:chosen.append(options[0])
    for i in order:
        if i not in chosen and evid.get(int(ev[i])):chosen.append(i)
        if len(chosen)>=5:break
    benign={'directed_transfer':'A mistaken call, misread hand strength, or ordinary value betting can create the same chip-flow pattern.',
            'soft_play':'Pot control, draw realization, board texture, or an opponent already being all-in may explain passivity.',
            'coordinated_isolation':'Independent strong ranges, normal steals, position, and weak third-party hands may explain the pressure.',
            'other_coordination':'A strategy shift, limited exposure, or an imperfect player baseline can produce a behavioral outlier.'}
    reviews=[];md=['# Five evidence case reviews — drafts for human verification','',
                   'These are model-selected investigation hypotheses, not private-label confirmations. Review actual traces before publication.','']
    for case_no,i in enumerate(chosen[:5],1):
        p=int(ev[i]);fam=str(behavior[i]);hands=evid[p]
        packets=[hand_packet(pack,pairs,p,h) for h in hands]
        observable_events=[event for packet in packets for event in observed_case_events(packet)]
        pair_actions=[a for packet in packets for a in packet['actions'] if a['role'] in ('A','B')]
        folds=sum(a['action']=='fold' for a in pair_actions)
        calls=sum(a['action']=='call' or (a['action']=='all_in' and a['amount_bb']<=a['to_call_bb']) for a in pair_actions)
        aggression=sum(a['action'] in ('bet','raise') or (a['action']=='all_in' and a['amount_bb']>a['to_call_bb']) for a in pair_actions)
        summary=f'Across the {len(hands)} submitted hands, the two players make {folds} folds, {calls} calls (including calling all-ins), and {aggression} aggressive actions. These counts describe the trace; they do not establish intent.'
        item=dict(case=case_no,pair_id=str(pairs.loc[p,'pair_id']),risk_score=float(risk[i]),predicted_behavior=fam,
                  evidence_hand_ids=[packet['hand_id'] for packet in packets],observable_summary=summary,
                  benign_alternative=benign.get(fam,'Ordinary independent play remains possible.'),observable_events=observable_events,hands=packets)
        reviews.append(item)
        md += [f'## Case {case_no}: {item["pair_id"]}',f'**Hypothesis:** {fam}. **Ranking score:** {risk[i]:.6f}.',
               '**Hands, ranked:** '+', '.join(item['evidence_hand_ids']),summary,
               '**Observable events:** '+(' '.join(observable_events[:3]) or 'No strict partnership-response event found; inspect the supplied complete traces rather than assuming coordination.'),
               '**Plausible benign alternative:** '+item['benign_alternative'],
               '**Review needed:** check who faced whom, decision-time cards and pot odds, side-pot eligibility, and the players’ normal behavior against other opponents.','']
    root=Path(cfg.output_dir)
    save_json(root/'case_reviews.json',reviews);(root/'case_reviews.md').write_text('\n'.join(md))
    return reviews

def optional_llm_audit(reviews,cfg):
    target=Path(cfg.output_dir)/'llm_audit_status.json'
    if not cfg.llm_gguf_path:
        save_json(target,{'status':'DISABLED','submission_modified':False,'reason':'No local model configured; default CPU scoring is entirely non-LLM.'});return
    path=Path(cfg.llm_gguf_path)
    if not path.is_file():
        save_json(target,{'status':'SKIPPED_MODEL_NOT_FOUND','submission_modified':False});return
    try:
        from llama_cpp import Llama
    except ImportError:
        save_json(target,{'status':'SKIPPED_LLAMA_CPP_NOT_INSTALLED','submission_modified':False});return
    model=None;results=[]
    try:
        model=Llama(model_path=str(path),n_ctx=4096,n_threads=cfg.threads,n_gpu_layers=0,seed=cfg.seed,verbose=False)
        model_hash=sha256_file(path)
        for review in reviews[:cfg.llm_cases]:
            packet={'hypothesis':review['predicted_behavior'],'hand':review['hands'][0]}
            prompt=('Audit this synthetic poker trace. IDs are references only. Do not invent actions or equate a made-hand rank with equity. '
                    'Return one JSON object with keys summary (string), benign_alternative (string), action_nos (array of integers), '
                    'and confidence (number between 0 and 1). Cite only action numbers present in this hand. '
                    'This is a review hypothesis, not proven coordination. /no_think\n'+json.dumps(packet))
            key=hashlib.sha256((model_hash+prompt).encode()).hexdigest()[:20]
            cache=Path(cfg.output_dir)/'llm_cache'/f'{key}.json';cache.parent.mkdir(exist_ok=True)
            if cache.exists():results.append(json.loads(cache.read_text()));continue
            answer=model.create_chat_completion(messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=cfg.llm_max_tokens)
            text=answer['choices'][0]['message']['content'] or ''
            # No eval, exec, shell invocation, or model-authored code execution.
            start=text.find('{');end=text.rfind('}')
            parsed=json.loads(text[start:end+1])
            valid_actions={a['action_no'] for a in packet['hand']['actions']}
            assert isinstance(parsed.get('summary'),str) and isinstance(parsed.get('benign_alternative'),str)
            assert isinstance(parsed.get('action_nos'),list) and all(type(n) is int for n in parsed['action_nos']) and set(parsed['action_nos']).issubset(valid_actions)
            assert 0<=float(parsed.get('confidence',-1))<=1
            row={'pair_id':review['pair_id'],'hand_id':packet['hand']['hand_id'],'audit':parsed,'status':'REFERENCE_VALIDATED_NOT_FACT_VERIFIED'}
            save_json(cache,row);results.append(row)
        save_json(target,{'status':'COMPLETED','submission_modified':False,'results':results})
    except Exception as exc:
        save_json(target,{'status':'AUDIT_FAILED_CLOSED','submission_modified':False,'error':repr(exc),'partial_results':results})
    finally:
        if model is not None:
            try:model.close()
            except Exception:pass



--- CODE CELL 34 ---
def capture_running_source():
    """Archive the executed implementation without downloading or executing external code."""
    try:
        if '__file__' in globals() and Path(__file__).is_file():
            text=Path(__file__).read_text()
        else:
            ip=get_ipython()
            cells=[c for c in ip.history_manager.input_hist_raw[1:] if c.strip() and not c.lstrip().startswith('# EXECUTE_PIPELINE')]
            text='\n\n'.join(cells)+"\n\nif __name__ == '__main__':\n    RESULT = run_pipeline(CFG)\n"
        (OUT/'pipeline_source.py').write_text(text)
    except Exception as exc:
        save_json(OUT/'source_export_status.json',{'status':'SOURCE_EXPORT_UNAVAILABLE','error':str(exc),
            'reproduction':'Keep the original executed notebook or supplied Python source.'})


def merge_stages(stages,n):
    out=dict(learned=np.zeros((n,len(LEARNED_FEATURES)),np.float32),aug_learned=np.zeros((n,len(LEARNED_FEATURES)),np.float32),
             aug_raw=np.zeros((n,len(PAIR_FEATURES)),np.float32),files=[])
    seen=set()
    for stage in stages:
        ids=[]
        for path in stage['files']:
            with np.load(path) as z:ids.extend(z['pids'].astype(int).tolist())
        if seen&set(ids):raise ValueError('Duplicate OOF feature rows.')
        seen.update(ids);ids=np.asarray(ids,np.int32)
        for key in ('learned','aug_learned','aug_raw'):
            if len(stage[key]):out[key][ids]=stage[key][ids]
        out['files']+=stage['files']
    return out




def average_pair_models(models,X):
    pred=None
    for model in models:
        p=model.predict(X)
        if pred is None:pred={k:np.asarray(v,float).copy() for k,v in p.items()}
        else:
            for k,v in p.items():pred[k]+=v
    for k in pred:pred[k]/=len(models)
    pred['other']=pred['other']>.5
    return pred


def predicted_behaviors(pred):
    behavior=FAMILIES[np.argmax(pred['family'],axis=1)].astype(object)
    behavior[pred['other']]='other_coordination'
    # No quota-based top-5% cut. All rows can provide evidence and retain continuous risk.
    return behavior


def evidence_from_scores(files,pairs,pred,ids,policy,gold=None,anchor_only=False):
    lookup={int(p):i for i,p in enumerate(ids)};evid={}
    for path in files:
        with np.load(path) as z:
            starts=np.r_[0,np.cumsum(z['counts'])]
            for j,p in enumerate(z['pids']):
                i=lookup.get(int(p))
                if i is None:continue
                a,b=starts[j:j+2]
                record=dict(hid=z['hid'][a:b],c=z['c'][a:b],r=z['r'][a:b],m=z['m'][a:b],rule=z['rule'][a:b],g=z['g'][a:b],d=z['d'][a:b],q=z['q'][a:b],t=z['t'][a:b],w=z['w'][a:b])
                evid[int(p)]=policy.rank(record,pred['family'][i],bool(pred['other'][i]),anchor_only=anchor_only)
    return evid



def evidence_diagnostics(records,pairs,pred,ids,policy):
    lookup={int(p):i for i,p in enumerate(ids)};values=[];per=defaultdict(list);recall=[];oracle=[];expert=[]
    for r in records:
        p=r['pid'];i=lookup[p];fam=int(np.flatnonzero(FAMILIES==pairs.loc[p,'family'])[0])
        val=ap5(policy.rank(r,pred['family'][i],bool(pred['other'][i])),r['gold'])
        perfect=np.eye(3)[fam];values.append(val);per[str(FAMILIES[fam])].append(val)
        oracle.append(ap5(policy.rank(r,perfect),r['gold']))
        recall.append(len(set(r['hid'].tolist())&r['gold'])/max(1,len(r['gold'])))
        ex=evidence_experts(r['c'],r['r'],r['rule'],fam,r['m'],r.get('g'),r.get('d'))
        expert.append([ap5(r['hid'][np.argsort(-ex[:,j],kind='stable')[:5]].tolist(),r['gold']) for j in range(len(EXPERT_NAMES))])
    new_scores=[];anchor_scores=[]
    for r in records:
        p=r['pid'];i=lookup[p];f=int(np.flatnonzero(FAMILIES==pairs.loc[p,'family'])[0])
        ee=challenger_evidence_experts(r,f)
        new_scores.append([ap5(r['hid'][np.argsort(-ee[:,j],kind='stable')[:5]].tolist(),r['gold']) for j in range(4)])
        anchor_scores.append(ap5(policy.rank(r,pred['family'][i],bool(pred['other'][i]),anchor_only=True),r['gold']))
    return dict(new_expert_names=NEW_EXPERT_NAMES,new_expert_map5=np.mean(new_scores,axis=0).tolist() if new_scores else [],
                anchor_evidence_map5=float(np.mean(anchor_scores)) if anchor_scores else 0.,queries=len(values),candidate_recall=float(np.mean(recall)) if recall else 0.,
                evidence_map5=float(np.mean(values)) if values else 0.,
                true_family_routing_map5=float(np.mean(oracle)) if oracle else 0.,
                per_family={f:float(np.mean(v)) for f,v in per.items()},
                expert_map5=np.mean(expert,axis=0).tolist() if expert else [],
                expert_names=EXPERT_NAMES,
                scan='ALL_SHARED_HANDS; NO_GOLD_INSERTION; NO_SCORE_CUTOFF')




--- CODE CELL 36 ---
def temporal_tail_audit(pack,pairs,paths,experts,pair_models,policy,lock,gold,cfg):
    target=lock&(pairs.label.to_numpy()>=0)
    stage=score_blocks(experts,paths,pairs,target,OUT/'outer_short_view',cfg,tail_fraction=2/3)
    ids=np.flatnonzero(target);X=np.column_stack([stage['view_raw'][ids],stage['learned'][ids]])
    pred=average_pair_models(pair_models,X);behavior=predicted_behaviors(pred)
    kept={}
    for path in stage['files']:
        with np.load(path) as z:
            bounds=np.r_[0,np.cumsum(z['counts'])]
            for j,p in enumerate(z['pids']):kept[int(p)]=set(z['hid'][bounds[j]:bounds[j+1]].tolist())
    tail_gold={int(p):gold.get(int(p),set())&kept.get(int(p),set()) for p in ids}
    frame=pairs.loc[ids].copy();excluded=0
    for p in ids:
        if pairs.loc[p,'label']==1 and not tail_gold[int(p)]:frame.loc[p,'label']=-1;excluded+=1
    evidence=evidence_from_scores(stage['files'],pairs,pred,ids,policy)
    report=metric_proxy(frame,pred['risk'],behavior,evidence,tail_gold)
    report.update(positive_pairs_without_visible_gold_excluded=excluded,
        status='FROZEN_OUTER_DIAGNOSTIC_ONLY; active-labelled-positive subset, not evaluation prevalence',
        query_context_recomputed=True,ranking_predictions_recomputed=True,
        historical_context='Personal action reference uses the full observed phase, always excluding all shared hands; no target labels enter this reference.')
    if report.get('n_positive',0)==0:
        for key in ('pair_ap','evidence_map5','behavior_map','composite_proxy','sklearn_pair_ap'):report[key]=None
        report['status']='NOT_ESTIMABLE_NO_ACTIVE_ANNOTATED_POSITIVES_IN_TAIL; labels were filtered, not inferred'
    save_json(OUT/'outer_temporal_tail_report.json',report)
    return report



def pair_gate_utility(y,risk,ids):
    known=y>=0
    return (.50*ranked_ap(y[known],risk[known],ids[known])+
            .25*policy_weighted_ap(y,risk,ids,.005)+.25*policy_weighted_ap(y,risk,ids,.02))

def select_nested_pair_challenger(pack,pairs,rawX,paths,data,gold,folds,tune,gate_stage,cfg):
    global NESTED_SELECTION_FILES
    gatefold=cfg.n_folds-1
    trainmask=tune&(folds!=gatefold);validmask=tune&(folds==gatefold)
    cfg.v6_pair_weight=0.
    if not cfg.nested_pair_selection:
        report=dict(selected_weight=0.,status='DISABLED; anchor risk kept')
        save_json(OUT/'v6_nested_pair_gate.json',report);return report
    # No upstream hand model generating the inner training features sees gatefold.
    stages=[];lineage=[]
    gate_pools=set(pairs.loc[validmask,'pool'].astype(int))
    for f in np.unique(folds[trainmask]):
        tr=trainmask&(folds!=f);va=trainmask&(folds==f)
        hand=HandExperts(cfg,cfg.seed+9100+int(f)).fit(data,tr,pairs)
        assert hand.training_pools.isdisjoint(gate_pools),'Nested upstream gate-label leakage'
        lineage.append(dict(scored_fold=int(f),training_pools=sorted(hand.training_pools),gate_pool_overlap=0))
        stages.append(score_blocks([hand],paths,pairs,va,OUT/f'nested_pair_train_{int(f)}',cfg))
        del hand;gc.collect()
    stage=merge_stages(stages,len(pairs));NESTED_SELECTION_FILES=list(stage['files']);trainids=np.flatnonzero(trainmask);validids=np.flatnonzero(validmask)
    X,y,fam,wt,_=augmented_training(pairs,trainids,rawX,stage,gold)
    model=PairModel(cfg,cfg.seed+9411).fit(X,y,fam,wt)
    vx=np.column_stack([rawX[validids],gate_stage['learned'][validids]])
    p=model.predict(vx);yy=pairs.loc[validids,'label'].to_numpy();ids=pairs.loc[validids,'pair_id'].to_numpy()
    pool=pairs.loc[validids,'pool'].to_numpy();unique=np.unique(pool)
    base=p['risk_anchor'];candidate=p['risk_challenger'];anchor_utility=pair_gate_utility(yy,base,ids)
    rng=np.random.default_rng(cfg.seed+9431)
    rows=[]
    for alpha in (.15,.30,.50):
        risk=(1-alpha)*base+alpha*candidate;utility=pair_gate_utility(yy,risk,ids)
        draws=[]
        for _ in range(240):
            selected=rng.choice(unique,len(unique),replace=True)
            ix=np.concatenate([np.flatnonzero(pool==v) for v in selected])
            if len(np.unique(yy[ix][yy[ix]>=0]))<2:continue
            draws.append(pair_gate_utility(yy[ix],risk[ix],ids[ix])-pair_gate_utility(yy[ix],base[ix],ids[ix]))
        lower=float(np.quantile(draws,.05/3)) if draws else -1.
        admitted=bool(len(unique)>=cfg.gate_min_pools and (yy==1).sum()>=cfg.gate_min_queries and
                      utility-anchor_utility>.001 and lower>0)
        rows.append(dict(weight=alpha,utility=utility,gain=utility-anchor_utility,
                         pool_bootstrap_lower=lower,admitted=admitted))
    accepted=[row for row in rows if row['admitted']]
    chosen=max(accepted,key=lambda row:row['utility'])['weight'] if accepted else 0.
    cfg.v6_pair_weight=float(chosen)
    report=dict(selected_weight=chosen,anchor_utility=anchor_utility,candidates=rows,gate_fold=int(gatefold),
                gate_pools=len(unique),gate_positives=int((yy==1).sum()),upstream_lineage=lineage,
                status='NESTED INNER HOLDOUT; no gate-pool labels in upstream meta-training features; outer fold 0 untouched',
                caveat='Prevalence values are sensitivity assumptions. The unknown family remains unvalidated.')
    save_json(OUT/'v6_nested_pair_gate.json',report);log('nested_pair_policy_frozen',weight=chosen)
    del X,vx,model,stages,stage;gc.collect()
    return report



--- CODE CELL 38 ---
def train_and_validate(pack,pairs,rawX,paths,cfg):
    gold=evidence_gold_map(pack,pairs);data=load_evidence_training(paths,pairs,gold)
    folds=assign_folds(pairs,cfg);pairs['fold']=folds
    pairs[['pair_id','phase','pool','fold','label']].to_csv(OUT/'validation_splits.csv',index=False)
    development=pairs.phase.to_numpy()==0;lock=development&(folds==0);tune=development&(folds>0)
    pair_policy_report=select_pair_policy(rawX,pairs,tune,folds,cfg) if cfg.pair_policy_search else {"status":"DISABLED"}
    inner=[];evidence_models=[]
    for f in range(1,cfg.n_folds):
        train=tune&(folds!=f);valid=tune&(folds==f)
        model=HandExperts(cfg,cfg.seed+f*113).fit(data,train,pairs)
        stage=score_blocks([model],paths,pairs,valid,OUT/f'inner_scores_{f}',cfg)
        inner.append(stage);evidence_models.append(model)
        log('inner_evidence_crossfit_complete',fold=f,train_pairs=int(train.sum()),valid_pairs=int(valid.sum()))
    tune_stage=merge_stages(inner,len(pairs))
    # Rebuild search-fold hand predictions without any gate labels.
    gate_stage=inner[-1]
    nested_report=select_nested_pair_challenger(pack,pairs,rawX,paths,data,gold,folds,tune,gate_stage,cfg)
    selection_files=list(NESTED_SELECTION_FILES)+list(gate_stage['files'])
    inner_records=read_records(selection_files,pairs,gold)
    routing=raw_routing_oof(rawX,pairs,tune,folds,cfg)
    strict_search=tune&(folds!=cfg.n_folds-1)
    routing.update(raw_routing_oof(rawX,pairs,strict_search,folds,cfg))
    for record in inner_records:
        if record['pid'] in routing:record['routing']=routing[record['pid']]
    policy=EvidencePolicy().fit(inner_records,pairs)
    del gate_stage,inner;gc.collect()
    save_json(OUT/'v6_evidence_gate.json',dict(challenger_weights=policy.challenger.tolist(),gates=policy.gate_report,
        search_folds=list(range(1,cfg.n_folds-1)),gate_fold=cfg.n_folds-1,outer_fold=0))
    save_json(OUT/'frozen_evidence_policy.json',dict(weights=policy.weights.tolist(),tuning=policy.report,probability_power=policy.power,routing=policy.routing_report,
       status='Selected only on inner pool-OOF annotations; frozen before outer pool scoring.',
       rl_scope='Finite expert-policy reward optimization, not online RL.'))
    trainids=np.flatnonzero(tune)
    X,y,fam,w,augids=augmented_training(pairs,trainids,rawX,tune_stage,gold)
    meta=[PairModel(cfg,seed).fit(X,y,fam,w) for seed in cfg.bag_seeds]
    del X,tune_stage,inner_records,routing;gc.collect()
    lock_stage=score_blocks(evidence_models,paths,pairs,lock,OUT/'outer_pool_scores',cfg,need_augmentation=False)
    valids=np.flatnonzero(lock);vx=np.column_stack([rawX[valids],lock_stage['learned'][valids]])
    pred=average_pair_models(meta,vx);behavior=predicted_behaviors(pred)
    evid=evidence_from_scores(lock_stage['files'],pairs,pred,valids,policy,gold)
    report=metric_proxy(pairs.loc[valids],pred['risk'],behavior,evid,gold)
    report['pair_policy']=pair_policy_report
    report['v6_nested_pair_gate']=nested_report
    report['v6_evidence_gate']=policy.gate_report
    anchor_evid=evidence_from_scores(lock_stage['files'],pairs,pred,valids,policy,gold,anchor_only=True)
    report['same_run_anchor']=metric_proxy(pairs.loc[valids],pred['risk_anchor'],behavior,anchor_evid,gold)
    report['v6_vs_anchor_composite_delta']=report['composite_proxy']-report['same_run_anchor']['composite_proxy']
    report['action_reference']='Own current-street context; all shared-hand actor counts excluded; observational, not causal.'
    if cfg.temporal_tail_audit:
        report['short_view_diagnostic']=temporal_tail_audit(pack,pairs,paths,evidence_models,meta,policy,lock,gold,cfg)
    report['retrieval']=evidence_diagnostics(read_records(lock_stage['files'],pairs,gold),pairs,pred,valids,policy)
    report['selected_negative_prevalence_stress']=prevalence_stress(pairs.loc[valids,'label'].to_numpy(),pred['risk'],pairs.loc[valids,'pair_id'].to_numpy())
    report['prevalence_stress_interpretation']='Reweighted confirmed negatives, NOT estimated evaluation AP.'
    report['positive_vs_background_stress_ap']=ranked_ap((pairs.loc[valids,'label'].to_numpy()==1).astype(int),pred['risk'],pairs.loc[valids,'pair_id'].to_numpy())
    report['background_stress_interpretation']='Unknowns treated as non-annotated background ONLY for sensitivity analysis; not ground truth, not a lower bound.'
    report['pool_bootstrap_95pct']=pool_bootstrap_interval(pairs,valids,pred['risk'],behavior,evid,gold,100 if pack.fixture else 250)
    # Same holdout, same evidence policy: measures the incremental learned-evidence pair features.
    baseline=PairModel(cfg,cfg.seed).fit(rawX[trainids],pairs.loc[trainids,'label'].to_numpy(),pairs.loc[trainids,'family'].to_numpy())
    bp=baseline.predict(rawX[valids])
    report['raw_only_pair_ap_ablation']=ranked_ap(pairs.loc[valids,'label'].to_numpy()[pairs.loc[valids,'label'].to_numpy()>=0],
        bp['risk'][pairs.loc[valids,'label'].to_numpy()>=0],pairs.loc[valids,'pair_id'].to_numpy()[pairs.loc[valids,'label'].to_numpy()>=0])
    if cfg.novelty_audit:report['unseen_family_diagnostic']=leave_one_family_out_audit(rawX,pairs,tune,lock,cfg)
    negmask=pairs.loc[valids,'label'].to_numpy()==0
    false_alerts=pd.DataFrame({'pair_id':pairs.loc[valids[negmask],'pair_id'].to_numpy(),'risk':pred['risk'][negmask],'predicted_behavior':behavior[negmask]})
    false_alerts.sort_values('risk',ascending=False).head(30).to_csv(OUT/'top_confirmed_non_target_alerts.csv',index=False)
    report['status']='OUTER_POOL_EVALUATION; no model/policy retuning on this report'
    report['upstream_label_leakage_checks']='Each scored pool disjoint from every upstream evidence model training pool.'
    report['public_private_labels']='Private labels unavailable; public labels are selected positives/confirmed non-targets only.'
    report['official_metric']='NOT_RUN: public-spec proxy; exact reference implementation was not retrieved.'
    report['full_competition_run']='FIXTURE_ONLY' if pack.fixture else 'COMPLETED_IN_THIS_RUNTIME'
    report['leaderboard_score']='NOT_SUBMITTED_BY_NOTEBOOK'
    save_json(OUT/'validation_report.json',report)
    portfolio_outer_artifacts(pack,pairs,valids,pred,evid,anchor_evid,gold,report)
    log('outer_pool_result',pair_ap=report['pair_ap'],evidence_map5=report['evidence_map5'],behavior_map=report['behavior_map'],composite_proxy=report['composite_proxy'],candidate_recall=report['retrieval']['candidate_recall'],background_stress=report['positive_vs_background_stress_ap'])
    pd.DataFrame({'pair_id':pairs.loc[valids,'pair_id'].to_numpy(),'label':pairs.loc[valids,'label'].to_numpy(),
                  'risk':pred['risk'],'predicted_behavior':behavior}).to_csv(OUT/'outer_predictions.csv',index=False)
    # Refit stage: now crossfit evidence on ALL development pools for unbiased meta training inputs.
    # Outer evaluation above remains independent; no weight updates use its outcomes.
    all_stages=[]
    for f in range(cfg.n_folds):
        train=development&(folds!=f);valid=development&(folds==f)
        model=HandExperts(cfg,cfg.seed+911+f*31).fit(data,train,pairs)
        stage=score_blocks([model],paths,pairs,valid,OUT/f'refit_oof_{f}',cfg)
        all_stages.append(stage)
        log('final_evidence_oof_complete',fold=f)
        del model;gc.collect()
    final_stage=merge_stages(all_stages,len(pairs))
    fullids=np.flatnonzero(development)
    X,y,fam,w,augids=augmented_training(pairs,fullids,rawX,final_stage,gold)
    pairmodels=[PairModel(cfg,seed+19).fit(X,y,fam,w) for seed in cfg.bag_seeds]
    del X,all_stages,meta,evidence_models;gc.collect()
    handmodels=[HandExperts(cfg,seed+1223).fit(data,development,pairs) for seed in cfg.bag_seeds]
    # Retain policy selected before outer scoring; do not refit policy to outer results.
    models=dict(pair=pairmodels,hand=handmodels,policy=policy,
                pair_features=PAIR_FEATURES+LEARNED_FEATURES,evidence_features=EVIDENCE_FEATURES,
                config=asdict(cfg),version=PIPELINE_VERSION)
    save_json(OUT/'hand_supervision.json',[m.supervision_report for m in handmodels])
    joblib.dump(models,OUT/'models.joblib',compress=3)
    return models,report,gold






--- CODE CELL 40 ---
def infer_all_pairs(pack,pairs,rawX,paths,models,cfg):
    mask=pairs.phase.to_numpy()==1;ids=np.flatnonzero(mask)
    # Evaluation pools can match training pool identifiers; players, activities and phases differ.
    # The models never receive pool IDs. The public split only guarantees positive-player exclusions.
    stage=score_blocks(models['hand'],paths,pairs,mask,OUT/'evaluation_scores',cfg,expected_disjoint=False,need_augmentation=False)
    X=np.column_stack([rawX[ids],stage['learned'][ids]])
    pred=average_pair_models(models['pair'],X);behavior=predicted_behaviors(pred)
    evid=evidence_from_scores(stage['files'],pairs,pred,ids,models['policy'])
    if cfg.keep_anchor_csv:
        ae=evidence_from_scores(stage['files'],pairs,pred,ids,models['policy'],anchor_only=True)
        from dataclasses import replace
        anchor_cfg=replace(cfg,submission_path=str(OUT/'anchor_comparison_DO_NOT_SUBMIT_BY_DEFAULT.csv'))
        _,anchor_path=write_validated_submission(pack,pairs,ids,pred['risk_anchor'],behavior,ae,anchor_cfg)
        save_json(OUT/'anchor_comparison.json',dict(path=str(anchor_path),risk='same-run protected V4 feature/model path',
            evidence='V4-style control selected on all inner OOF records; NOT a byte-identical historical V4 submission',
            leaderboard_score='UNMEASURED'))
    log('evaluation_complete',pairs=len(ids),pairs_with_evidence=sum(bool(evid.get(int(p))) for p in ids),
        hand_scan='ALL_SHARED_HANDS',behavior_counts=pd.Series(behavior).value_counts().to_dict())
    return ids,pred['risk'],behavior,evid





def write_validated_submission(pack,pairs,ids,risk,behavior,evid,cfg):
    destination=Path(cfg.submission_path)
    if destination.suffix.lower()!='.csv':raise ValueError('Submission destination must end in .csv, never .json.')
    if pack.fixture:
        destination=Path(cfg.output_dir)/('fixture_anchor_DO_NOT_SUBMIT.csv' if 'anchor_comparison' in destination.name else 'fixture_submission_DO_NOT_SUBMIT.csv')
    destination.parent.mkdir(parents=True,exist_ok=True)
    expected=['pair_id','risk_score','predicted_behavior']+[f'evidence_hand_{i}' for i in range(1,6)]
    if list(pack.sample.columns)!=expected:raise ValueError('Sample template changed; inspect columns explicitly.')
    if len(ids)!=len(pack.eval_pairs):raise ValueError('Missing/extra evaluation pairs.')
    if not np.isfinite(risk).all() or np.any((risk<0)|(risk>1)):raise ValueError('Invalid risk values.')
    if not set(behavior)<=ALLOWED_BEHAVIORS:raise ValueError('Invalid behavior.')
    output=pd.DataFrame({'pair_id':pairs.loc[ids,'pair_id'].to_numpy(),'risk_score':risk,'predicted_behavior':behavior})
    rows=[]
    for p in ids:
        hh=evid.get(int(p),[])
        if len(hh)>5 or len(set(hh))!=len(hh):raise ValueError('Duplicate/excess evidence.')
        for h in hh:
            if not 0<=h<len(pack.hand_ids) or pack.hand_values[h,5]!=1:raise ValueError('Non-evaluation evidence hand.')
            if pairs.loc[p,'u'] not in pack.seat_players[h] or pairs.loc[p,'v'] not in pack.seat_players[h]:raise ValueError('Non-shared evidence hand.')
        rows.append([str(pack.hand_ids[h]) for h in hh]+['NO_EVIDENCE']*(5-len(hh)))
    output[expected[3:]]=np.array(rows,object)
    output=pack.sample[['pair_id']].merge(output,on='pair_id',how='left',validate='one_to_one')[expected]
    if output.isna().any().any() or (output.astype(str)=='').any().any():raise ValueError('Empty submission cells.')
    temp=destination.with_name(destination.name+'.tmp');output.to_csv(temp,index=False,float_format='%.12g');temp.replace(destination)
    readback=pd.read_csv(destination,dtype={c:str for c in expected if c!='risk_score'})
    assert list(readback.columns)==expected and readback.pair_id.tolist()==pack.sample.pair_id.tolist()
    assert readback.shape==output.shape and not readback.isna().any().any()
    if not np.isfinite(readback.risk_score).all() or not readback.risk_score.between(0,1).all():raise ValueError('Serialization changed risk validity.')
    manifest=dict(file=str(destination),rows=len(output),sha256=sha256_file(destination),bytes=destination.stat().st_size,
                  extension='.csv',all_shared_hands_checked=True,all_evaluation_phases_checked=True,
                  duplicates_checked=True,exact_template_checked=True,fixture=pack.fixture)
    save_json(OUT/'submission_validation.json',manifest)
    log('CSV_READY_SELECT_THIS_FILE',**manifest)
    return output,destination





--- CODE CELL 42 ---
def write_diagnostics(models,report,pairs,ids,risk,behavior,cfg):
    from zipfile import ZipFile,ZIP_DEFLATED
    columns=models['pair_features']
    anchor_gain=np.mean([m.pu.booster_.feature_importance(importance_type='gain') for m in models['pair']],axis=0)
    gain=np.pad(anchor_gain,(0,len(columns)-len(anchor_gain)))
    challenge=[m.challenger_pu.booster_.feature_importance(importance_type='gain') for m in models['pair'] if m.challenger_pu is not None]
    if challenge:pd.DataFrame({'feature':columns,'gain':np.mean(challenge,axis=0)}).sort_values('gain',ascending=False).to_csv(OUT/'challenger_feature_importance.csv',index=False)
    pd.DataFrame({'feature':columns,'gain':gain}).sort_values('gain',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4));ax.hist(risk,bins=60)
        ax.set(xlabel='Uncalibrated risk ranking score',ylabel='Evaluation pairs',title='Score distribution')
        fig.tight_layout();fig.savefig(OUT/'risk_distribution.png',dpi=130);plt.close(fig)
        order=np.argsort(gain)[-20:];fig,ax=plt.subplots(figsize=(10,7))
        ax.barh(np.array(columns)[order],gain[order]);ax.set(xlabel='Gain',title='PU pair model features')
        fig.tight_layout();fig.savefig(OUT/'feature_importance.png',dpi=130);plt.close(fig)
    except ImportError:log('plots_skipped',reason='matplotlib unavailable')
    if cfg.archive_diagnostics and not cfg.fixture_mode:
        target=Path(cfg.submission_path).parent/'poker7_05_gated_consensus_DIAGNOSTICS_DO_NOT_SUBMIT.zip'
        include=['portfolio_method.json','portfolio_preflight.json','portfolio_outer_predictions.csv','portfolio_outer_evidence.json','portfolio_evaluation_predictions.csv','v6_evidence_gate.json','v6_nested_pair_gate.json','anchor_comparison.json','anchor_comparison_DO_NOT_SUBMIT_BY_DEFAULT.csv','hand_supervision.json','challenger_feature_importance.csv','preflight.json','environment.json','validation_report.json','validation_splits.csv','frozen_evidence_policy.json',
                 'outer_predictions.csv','top_confirmed_non_target_alerts.csv','frozen_pair_policy.json','outer_temporal_tail_report.json','submission_validation.json','run_manifest.json','stage_log.json',
                 'case_reviews.json','case_reviews.md','feature_importance.csv','risk_distribution.png',
                 'feature_importance.png','llm_audit_status.json','models.joblib','pipeline_source.py']
        with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=3) as z:
            for name in include:
                p=OUT/name
                if p.is_file():z.write(p,name)
        return target
    return None






--- CODE CELL 44 ---
def run_preflight(cfg):
    import xgboost
    major=int(xgboost.__version__.split('.')[0])
    if major<2:raise RuntimeError('XGBoost >=2 is required for the MAP top-k configuration. No silent install is performed.')
    assert len(ANCHOR_EVIDENCE_FEATURES)==289 and ANCHOR_PAIR_WIDTH==1123
    assert not cfg.llm_gguf_path or Path(cfg.llm_gguf_path).is_file()
    z=np.zeros((7,len(HAND_FEATURES)),np.float32);z[:,HF['progress']]=np.linspace(0,1,7)
    zz=evidence_matrix(z)
    assert np.isfinite(zz).all() and np.array_equal(zz[:,:289],anchor_evidence_matrix(z))
    assert np.isclose(ap5([1,2,3,4,5],{1,2,3,4,5}),1.)
    assert not paired_pool_gate(np.zeros(24),np.repeat(np.arange(8),3))['passed']
    save_json(OUT/'preflight.json',dict(status='PASS',anchor_columns=289,query_columns=len(QUERY_FEATURES),
        cpu_only=True,xgboost_version=xgboost.__version__,fixture_mode=cfg.fixture_mode,
        caveat='Preflight tests software contracts only, not leaderboard accuracy.'))

def run_pipeline(cfg=None,raw_frames=None):
    global CFG,OUT,RUN_START
    RUN_LOG.clear();RUN_START=time.perf_counter()
    cfg=cfg or CFG;CFG=cfg;OUT=Path(cfg.output_dir);OUT.mkdir(parents=True,exist_ok=True)
    if raw_frames is None and cfg.fixture_mode:raise RuntimeError('No synthetic-data fallback is permitted for competition runs.')
    if Path(cfg.submission_path).suffix.lower()!='.csv':raise ValueError('Set submission_path to submission.csv, not a JSON report.')
    if cfg.n_folds<5:raise ValueError('V6 requires at least five pool folds: outer, search and nested gate separation.')
    if not cfg.full_scan:raise ValueError('V6 requires full_scan=True; all shared hands must remain eligible.')
    if cfg.threads<1:raise ValueError('threads must be positive.')
    start=time.perf_counter();versions=check_dependencies()
    if cfg.run_preflight:run_preflight(cfg)
    capture_running_source()
    data_root=None if raw_frames is not None else discover_data(cfg.data_dir)
    with threadpool_limits(limits=cfg.threads):
        pack=prepare_pack(data_root,cfg,raw_frames)
        pairs=build_pairs(pack,cfg)
        pairs['u_local']=pack.player_local[pairs.u.to_numpy()];pairs['v_local']=pack.player_local[pairs.v.to_numpy()]
        rawX,paths=extract_features(pack,pairs,cfg)
        assert np.isfinite(rawX).all()
        forbidden=('player_id','pair_id','table_id','account_age','region','client')
        assert not any(any(k in f for k in forbidden) for f in PAIR_FEATURES+EVIDENCE_FEATURES+LEARNED_FEATURES)
        models,report,gold=train_and_validate(pack,pairs,rawX,paths,cfg)
        ids,risk,behavior,evid=infer_all_pairs(pack,pairs,rawX,paths,models,cfg)
        sub,destination=write_validated_submission(pack,pairs,ids,risk,behavior,evid,cfg)
        reviews=make_case_reviews(pack,pairs,ids,risk,behavior,evid,cfg);optional_llm_audit(reviews,cfg)
        save_json(OUT/'run_manifest.json',dict(version=PIPELINE_VERSION,configuration=asdict(cfg),environment=versions,
            runtime_code_fingerprint=runtime_source_fingerprint(),elapsed_seconds=time.perf_counter()-start,
            fixture=pack.fixture,submission=str(destination),submission_sha256=sha256_file(destination),
            public_score='NOT_RUN',target_score=.970,target_status='ASPIRATION_NOT_MEASURED',
            baseline_user_reported_score=.83065, v5_user_reported_score=.82757,official_metric='PUBLIC_SPEC_PROXY_ONLY'))
        save_json(OUT/'stage_log.json',RUN_LOG)
        archive=write_diagnostics(models,report,pairs,ids,risk,behavior,cfg)
    log('run_complete',submission=str(destination),seconds_observed=round(time.perf_counter()-start,2),fixture=pack.fixture)
    return dict(submission=sub,submission_path=destination,diagnostics_archive=archive,report=report,models=models,
                pack=pack,pairs=pairs,features=rawX,evidence=evid,evaluation_indices=ids,risk=risk,behavior=behavior)




--- CODE CELL 46 ---
# Five predeclared experiments. Changing CFG.variant after definitions is forbidden.
PROFILE = CFG.variant
PROFILE_NAMES = {
    'precision': '01 Precision Rank',
    'partial': '02 Partial-Label Witness',
    'temporal': '03 Episode Scan',
    'robust': '04 Robust PU Bagging',
    'consensus': '05 Gated Consensus',
}
if PROFILE not in PROFILE_NAMES:
    raise ValueError('Unknown experiment profile: '+PROFILE)
USE_EPISODE_FEATURES = PROFILE in ('temporal','consensus')
USE_ROBUST_PAIR = PROFILE in ('robust','consensus')
V6_EVIDENCE_FEATURES = list(EVIDENCE_FEATURES)
_v6_evidence_matrix = evidence_matrix
EPISODE_SOURCES = ['rule_directed','rule_soft','rule_isolation','null_action_score',
    'partner_raise_then_fold','hu_strong_check_nonallin','null_partner_benefit_surprise',
    'null_outsider_benefit_surprise']
EPISODE_FEATURES = [f'episode_{name}_{stat}' for name in EPISODE_SOURCES for stat in
    ('neighbor3','neighbor9','neighbor21','own_vs_near','near_vs_background','neighbor_event_rate')]
if USE_EPISODE_FEATURES:
    EVIDENCE_FEATURES = V6_EVIDENCE_FEATURES + EPISODE_FEATURES


def leave_center_mean(values, width):
    """Centered retrospective context, excluding the decision being scored."""
    v=np.asarray(values,np.float64); n=len(v)
    if not n:return np.empty(0,np.float32)
    j=np.arange(n);a=np.maximum(0,j-width//2);b=np.minimum(n,j+width//2+1)
    cs=np.r_[0.,np.cumsum(v)]
    count=b-a-1
    return np.divide(cs[b]-cs[a]-v,count,out=np.zeros(n),where=count>0).astype(np.float32)


def evidence_matrix(x):
    base=_v6_evidence_matrix(x)
    if not USE_EPISODE_FEATURES:return base
    if not len(x):return np.empty((0,len(EVIDENCE_FEATURES)),np.float32)
    extra=[]
    for name in EPISODE_SOURCES:
        raw=np.maximum(0.,x[:,HF[name]].astype(np.float64));v=np.log1p(raw)
        near=leave_center_mean(v,9)
        extra += [leave_center_mean(v,w) for w in (3,9,21)]
        extra += [np.clip(v-near,-12,12),np.clip(near-np.median(v),-12,12),
                  leave_center_mean(raw>0,9)]
    result=np.column_stack([base,*extra]).astype(np.float32)
    if result.shape[1]!=len(EVIDENCE_FEATURES):raise AssertionError('Episode feature dimensions differ')
    return np.nan_to_num(result,nan=0.,posinf=0.,neginf=0.)


_v6_pair_features = evidence_pair_features
V6_LEARNED_FEATURES = list(LEARNED_FEATURES)
SCAN_STATS=['scan3_excess','scan9_excess','scan21_excess','run_length_fraction',
            'top5_mass_fraction','top5_time_span','top5_gap_cv','active_fraction']
if USE_EPISODE_FEATURES:
    LEARNED_FEATURES = V6_LEARNED_FEATURES + [f'episode_mil_{f}_{s}' for f in [*FAMILIES,'witness'] for s in SCAN_STATS]


def sequence_summary(v, progress):
    """Descriptive scan scores, NOT p-values or fitted hidden-state evidence."""
    v=np.clip(np.asarray(v,np.float64),0,1);n=len(v)
    if not n:return np.zeros(len(SCAN_STATS),np.float32)
    active=(v>=.5).astype(np.float64);p=(active.sum()+1)/(n+2);cs=np.r_[0.,active.cumsum()]
    scans=[]
    for width in (3,9,21):
        k=min(width,n);maximum=float(np.max(cs[k:]-cs[:-k]))
        scans.append(max(0.,(maximum-k*p)/np.sqrt(k*p*(1-p)+1.)-np.sqrt(2*np.log1p(n/k))))
    longest=0;run=0
    for a in active:
        run=run+1 if a else 0;longest=max(longest,run)
    best=np.argsort(-v,kind='stable')[:5]
    t=np.sort(np.asarray(progress,dtype=float)[best]);gaps=np.diff(t)
    cv=float(np.std(gaps)/(np.mean(gaps)+1e-6)) if len(gaps)>1 else 0.
    return np.asarray([*scans,longest/max(1,n),v[best].sum()/(v.sum()+1e-6),
                       np.ptp(t) if len(t)>1 else 0.,min(cv,10.),active.mean()],np.float32)


def evidence_pair_features(c,progress,q=None,w=None):
    old=_v6_pair_features(c,progress,q,w)
    if not USE_EPISODE_FEATURES:return old
    new=np.concatenate([sequence_summary(q[:,f],progress) for f in range(3)]+[sequence_summary(w,progress)])
    return np.r_[old,new].astype(np.float32)


--- CODE CELL 48 ---
def robust_margin_reference(scores):
    scores=np.asarray(scores,dtype=float)
    if not len(scores):return (0.,1.)
    q=np.quantile(scores,[.25,.5,.75]);return (float(q[1]),float(max(.20,q[2]-q[0])))


def reference_margin(scores, ref):
    return expit(np.clip((np.asarray(scores,dtype=float)-ref[0])/ref[1],-15,15)).astype(np.float32)


class HandExperts(AnchorHandExperts):
    """V4 anchor plus genuinely different, predeclared challenger learners."""
    def fit(self,data,mask,pairs):
        self.profile=self.cfg.variant
        anchor_data=dict(data);anchor_data['x']=data['x'][:,:len(ANCHOR_EVIDENCE_FEATURES)]
        super().fit(anchor_data,mask,pairs)
        use=mask[data['pid']]
        reference=use&np.isin(data['pid'],self.reference_pair_ids)
        train=use&~reference
        X=data['x'][train];y=data['y'][train];fam=data['family'][train]
        pp=data['pid'][train];neg=data['known_negative'][train];XC=data['x'][reference]
        _,inv,count=np.unique(pp,return_inverse=True,return_counts=True)
        qw=np.clip(np.median(count)/count[inv],.25,4.).astype(np.float32)
        self.query_models=[];self.query_refs=[];self.query_included_counts=[]
        self.query_witness_models=[];self.query_witness_refs=[]
        for f in range(3):
            target=((y==1)&(fam==f)).astype(np.int8)
            # Partial-label recipe never calls unannotated positive-pair hands benign.
            selected=((y==1)|neg) if self.profile=='partial' else np.ones(len(y),bool)
            weight=np.where(target,8.,np.where(neg,1.,np.where(y,.6,.10)))*np.sqrt(qw)
            par={**lgb_params(self.cfg,self.seed+7001+f,self.cfg.query_trees),
                 'num_leaves':11,'reg_lambda':16.,'min_child_samples':28}
            model=lgb.LGBMClassifier(**par)
            model.fit(X[selected],target[selected],sample_weight=weight[selected])
            self.query_models.append(model)
            self.query_refs.append(fit_score_reference(model.booster_.predict(XC,num_threads=self.cfg.threads)) if len(XC) else (0.,1.))
            self.query_included_counts.append(dict(family=str(FAMILIES[f]),rows=int(selected.sum()),
                unannotated_positive_hands_included=int((selected&(y==0)&~neg).sum())))
            if self.profile=='consensus':
                trusted=(y==1)|neg
                wm=lgb.LGBMClassifier(**{**par,'random_state':self.seed+7601+f,'reg_lambda':22.,
                                       'n_estimators':max(8,self.cfg.query_trees//2)})
                wm.fit(X[trusted],target[trusted],sample_weight=weight[trusted])
                self.query_witness_models.append(wm)
                self.query_witness_refs.append(fit_score_reference(wm.booster_.predict(XC,num_threads=self.cfg.threads)) if len(XC) else (0.,1.))
        # Every ranking group is an entire positive pair; the family is an explicit hypothesis.
        ix=np.flatnonzero(fam>=0);ix=ix[np.argsort(pp[ix],kind='stable')]
        _,groups=np.unique(pp[ix],return_counts=True)
        RX=np.column_stack([X[ix],np.eye(3,dtype=np.float32)[fam[ix]]]);ry=y[ix]
        if len(groups)<2:raise ValueError('Insufficient complete evidence-ranking queries')
        recipes={
            'precision':[('xgb','rank:map','topk',8),('xgb','rank:pairwise','mean',4)],
            'partial':[('lgb','lambdarank','topk',8),('xgb','rank:map','topk',8)],
            'temporal':[('xgb','rank:map','topk',12),('xgb','rank:ndcg','topk',8)],
            'robust':[('xgb','rank:map','topk',8)],
            'consensus':[('xgb','rank:map','topk',8),('xgb','rank:ndcg','topk',12),('xgb','rank:pairwise','mean',4)]
        }[self.profile]
        self.rank_bank=[];self.rank_recipes=[]
        import xgboost as xgb
        for j,(kind,obj,method,topk) in enumerate(recipes):
            rounds=max(8,int(np.ceil(self.cfg.map_trees/len(recipes))))
            if kind=='xgb':
                rank=xgb.XGBRanker(objective=obj,eval_metric='map@5',n_estimators=rounds,
                    learning_rate=.045,max_depth=4,min_child_weight=3.,reg_lambda=16.,reg_alpha=.08,
                    subsample=.90,colsample_bytree=.82,tree_method='hist',device='cpu',
                    n_jobs=self.cfg.threads,random_state=self.seed+7101+j*83,
                    lambdarank_pair_method=method,lambdarank_num_pair_per_sample=topk,verbosity=0)
                rank.fit(RX,ry,group=groups)
            else:
                rank=lgb.LGBMRanker(**{**lgb_params(self.cfg,self.seed+7101+j*83,rounds),
                    'objective':'lambdarank','metric':'map','label_gain':[0,1],
                    'lambdarank_truncation_level':8,'num_leaves':11,'reg_lambda':14.})
                # Weak annotation background, not a negative clinical/gameplay label.
                rank.fit(RX,ry,group=groups,sample_weight=np.where(ry==1,1.,.20))
            refs=[]
            for f in range(3):
                if len(XC):
                    h=np.zeros((len(XC),3),np.float32);h[:,f]=1.
                    xx=np.column_stack([XC,h])
                    scores=rank.predict(xx) if kind=='xgb' else rank.booster_.predict(xx,num_threads=self.cfg.threads)
                    refs.append(robust_margin_reference(scores))
                else:refs.append((0.,1.))
            self.rank_bank.append((kind,rank,refs))
            self.rank_recipes.append(dict(engine=kind,objective=obj,pair_method=method,topk=topk,trees=rounds))
        # Behavioral witness: annotated evidence vs confirmed non-target hands only.
        trusted=(y==1)|neg
        self.witness_indices=np.array([i for i,n in enumerate(EVIDENCE_FEATURES) if
            n.startswith(('null_','partner_','hu_','outsider_','voluntary_','query_','episode_','river_','rule_'))],np.int32)
        self.witness=lgb.LGBMClassifier(**{**lgb_params(self.cfg,self.seed+7201,self.cfg.witness_trees),
            'num_leaves':11,'reg_lambda':18.,'min_child_samples':24})
        wy=y[trusted]
        self.witness.fit(X[trusted][:,self.witness_indices],wy,
            sample_weight=np.where(wy,8.,1.)*np.sqrt(qw[trusted]))
        self.witness_reference=fit_score_reference(self.witness.booster_.predict(XC[:,self.witness_indices],num_threads=self.cfg.threads)) if len(XC) else (0.,1.)
        self.supervision_report=dict(profile=self.profile,fit_rows=int(train.sum()),reference_rows=int(reference.sum()),
            witness_positive=int(wy.sum()),witness_confirmed_non_target=int((wy==0).sum()),
            unannotated_positive_hands_excluded_from_witness=int(((y==0)&~neg).sum()),
            unknown_hands_in_witness=0,rank_queries=int(len(groups)),query_supervision=self.query_included_counts,
            rank_recipes=self.rank_recipes,reference_normalization='ranking scale reference; not a calibrated probability')
        return self
    def predict(self,X):return super().predict(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_likelihood(self,X):return super().predict_likelihood(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_generic(self,X):return super().predict_generic(X[:,:len(ANCHOR_EVIDENCE_FEATURES)])
    def predict_additions(self,X,with_map=True):
        if not len(X):return np.empty((0,3),np.float32),np.empty((0,3),np.float32),np.empty(0,np.float32)
        q=np.column_stack([normalize_score(m.booster_.predict(X,num_threads=self.cfg.threads),r)
            for m,r in zip(self.query_models,self.query_refs)]).astype(np.float32)
        if self.query_witness_models:
            qw=np.column_stack([normalize_score(m.booster_.predict(X,num_threads=self.cfg.threads),r)
                for m,r in zip(self.query_witness_models,self.query_witness_refs)])
            q=(.60*q+.40*qw).astype(np.float32)
        ranks=np.zeros((len(X),3),np.float32)
        if with_map:
            for f in range(3):
                h=np.zeros((len(X),3),np.float32);h[:,f]=1.;rx=np.column_stack([X,h])
                for kind,model,refs in self.rank_bank:
                    score=model.predict(rx) if kind=='xgb' else model.booster_.predict(rx,num_threads=self.cfg.threads)
                    ranks[:,f]+=reference_margin(score,refs[f])/len(self.rank_bank)
        w=normalize_score(self.witness.booster_.predict(X[:,self.witness_indices],num_threads=self.cfg.threads),self.witness_reference)
        return q,ranks,w


_V6PairModel=PairModel
class PairModel(_V6PairModel):
    """The original risk path is unchanged; PU-bank results are gated challengers."""
    def fit(self,X,y,family,exposure_weight=None):
        super().fit(X,y,family,exposure_weight)
        self.pu_bank=[];self.pu_bank_report=[]
        self.robust_profile=self.cfg.variant in ('robust','consensus')
        if not self.robust_profile or X.shape[1]<=ANCHOR_PAIR_WIDTH:return self
        pos=y==1;unknown=np.flatnonzero(y<0);known=np.flatnonzero(y>=0)
        mult=np.ones(len(y)) if exposure_weight is None else np.asarray(exposure_weight)
        teacher=self.challenger_pn.booster_.predict(X,num_threads=self.cfg.threads)
        par={**lgb_params(self.cfg,self.seed+8701,max(8,self.cfg.challenger_pair_trees//2)),
            'num_leaves':15,'min_child_samples':32,'reg_lambda':22.}
        # A fixed weak-background sensitivity bank, not estimated class priors or true negatives.
        for j,weak_weight in enumerate((.015,.04,.10)):
            rng=np.random.default_rng(self.seed+8729+j*101)
            chosen=rng.choice(unknown,size=max(1,int(np.ceil(.60*len(unknown)))),replace=False) if len(unknown) else unknown
            ids=np.r_[known,chosen];rng.shuffle(ids)
            weights=np.where(pos[ids],3.,np.where(y[ids]==0,1.,weak_weight))*mult[ids]
            weak=y[ids]<0
            weights[weak]*=(.15+.85*(1.-teacher[ids][weak]))
            model=lgb.LGBMClassifier(**{**par,'random_state':self.seed+8801+j*103})
            model.fit(X[ids],pos[ids].astype(np.int8),sample_weight=weights)
            self.pu_bank.append(model)
            self.pu_bank_report.append(dict(weak_background_weight=weak_weight,unknown_sampled=len(chosen),
                training_rows=len(ids),unknowns_are_not_confirmed_negatives=True))
        return self
    def predict(self,X):
        p=super().predict(X)
        if not self.pu_bank:return p
        scores=np.column_stack([m.booster_.predict(X,num_threads=self.cfg.threads) for m in self.pu_bank])
        pbar=scores.mean(axis=1);spread=scores.std(axis=1)
        pn=self.challenger_pn.booster_.predict(X,num_threads=self.cfg.threads)
        # A small disagreement penalty is fixed a priori. All scores remain ranking scores.
        candidate=np.clip(.25*pn+.75*pbar-.05*spread,1e-12,1-1e-12)
        p['risk_challenger']=candidate
        p['risk']=np.clip((1-self.v6_weight)*p['risk_anchor']+self.v6_weight*candidate,1e-12,1-1e-12)
        return p


--- CODE CELL 50 ---
NEW_EXPERT_NAMES = {
    'precision':['query_likelihood','MAP_RankNet_bank','trusted_witness','precision_consensus'],
    'partial':['family_witness','weak_annotation_rank_bank','trusted_witness','partial_label_consensus'],
    'temporal':['episode_query','episode_MAP_NDCG_bank','trusted_witness','bounded_episode_support'],
    'robust':['query_likelihood','pooled_MAP','trusted_witness','query_MAP_consensus'],
    'consensus':['dual_supervision_query','MAP_NDCG_RankNet_bank','trusted_witness','agreement_shrunk_consensus'],
}[PROFILE]


def challenger_evidence_experts(record,f):
    for k in ('q','t','w','m'):
        if k not in record:raise ValueError('Missing portfolio score channel: '+k)
    q=percentiles(record['q'][:,f]);t=record['t'][:,f];w=percentiles(record['w'])
    if PROFILE=='temporal':
        own=np.asarray(record['q'][:,f],np.float32)
        combo=percentiles(own*(.8+.2*episode_posterior(own)))
    elif PROFILE=='partial':combo=.5*q+.3*t+.2*w
    elif PROFILE=='consensus':
        panel=np.column_stack([q,t,w,record['m'][:,f]])
        combo=panel@np.array([.25,.40,.10,.25])-.08*panel.std(axis=1)
    else:combo=.45*q+.55*t
    return np.column_stack([q,t,w,combo])


def portfolio_outer_artifacts(pack,pairs,ids,pred,evid,anchor_evid,gold,report):
    """Audit output only; never consumed by training or automatic model selection."""
    df=pairs.loc[ids,['pair_id','pool','fold','label','family']].copy()
    df['risk_score']=pred['risk'];df['anchor_risk_score']=pred['risk_anchor']
    df['predicted_behavior']=predicted_behaviors(pred)
    for f,name in enumerate(FAMILIES):df['score_'+name]=pred['family'][:,f]
    rows=[]
    for p in ids:
        hh=evid.get(int(p),[]);gg=gold.get(int(p),set())
        rows.append(dict(pair_id=str(pairs.loc[p,'pair_id']),
            submitted_hands=[str(pack.hand_ids[h]) for h in hh],
            anchor_hands=[str(pack.hand_ids[h]) for h in anchor_evid.get(int(p),[])],
            gold_hands=[str(pack.hand_ids[h]) for h in sorted(gg)],
            ap5=ap5(hh,gg) if pairs.loc[p,'label']==1 else None))
    df.to_csv(OUT/'portfolio_outer_predictions.csv',index=False)
    save_json(OUT/'portfolio_outer_evidence.json',rows)
    report['portfolio_profile']=PROFILE
    report['portfolio_design']='Five predeclared standalone experiments; no claimed leaderboard gain.'
    report['evidence_selection_lineage']='Search scores come from nested models excluding both gate pools and their own scored pools.'
    report['shared_outer_holdout_warning']='Choosing among the five experiments with this outer score consumes it as a selection set; report all five, not just the winner.'
    save_json(OUT/'validation_report.json',report)


_old_infer_all_pairs=infer_all_pairs

def infer_all_pairs(pack,pairs,rawX,paths,models,cfg):
    ids,risk,behavior,evid=_old_infer_all_pairs(pack,pairs,rawX,paths,models,cfg)
    # Only a small, easily compared predictions file; no changes to submission risk.
    out=pd.DataFrame({'pair_id':pairs.loc[ids,'pair_id'].to_numpy(),
                      'risk_score':risk,'predicted_behavior':behavior})
    for j in range(5):
        out[f'evidence_hand_{j+1}']=[str(pack.hand_ids[evid[int(p)][j]]) if len(evid.get(int(p),[]))>j else 'NO_EVIDENCE' for p in ids]
    out.to_csv(OUT/'portfolio_evaluation_predictions.csv',index=False)
    save_json(OUT/'portfolio_method.json',dict(profile=PROFILE,name=PROFILE_NAMES[PROFILE],
        reference_source='uploaded Poker Sentinel V6; V4 anchor columns retained',
        raw_hand_features=len(HAND_FEATURES),anchor_evidence_features=len(ANCHOR_EVIDENCE_FEATURES),
        challenger_evidence_features=len(EVIDENCE_FEATURES),
        anchor_pair_features=ANCHOR_PAIR_WIDTH,challenger_pair_features=len(PAIR_FEATURES)+len(LEARNED_FEATURES),
        episode_features_enabled=USE_EPISODE_FEATURES,robust_pu_enabled=USE_ROBUST_PAIR,
        weak_pu_banks=[m.pu_bank_report for m in models['pair']],
        leaderboard_score='NOT_MEASURED',historical_user_reported_control=.83065))
    return ids,risk,behavior,evid


_old_preflight=run_preflight

def run_preflight(cfg):
    if cfg.variant!=PROFILE:raise ValueError('Change the configuration before running all cells, not after feature definitions.')
    if not cfg.nested_pair_selection:raise ValueError('This suite requires nested_pair_selection=True for uncontaminated search predictions.')
    if cfg.variant not in PROFILE_NAMES:raise ValueError('Unsupported profile')
    if cfg.map_trees<8:raise ValueError('map_trees must be at least 8')
    _old_preflight(cfg)
    z=np.zeros((17,len(HAND_FEATURES)),np.float32);z[:,HF['progress']]=np.linspace(0,1,len(z))
    matrix=evidence_matrix(z)
    assert matrix.shape==(len(z),len(EVIDENCE_FEATURES)) and np.isfinite(matrix).all()
    assert np.array_equal(matrix[:,:289],anchor_evidence_matrix(z))
    v=np.zeros(11);v[5]=1
    assert leave_center_mean(v,9)[5]==0
    c=np.zeros((17,3),np.float32);w=np.zeros(17,np.float32)
    features=evidence_pair_features(c,z[:,HF['progress']],c,w)
    assert len(features)==len(LEARNED_FEATURES) and np.isfinite(features).all()
    save_json(OUT/'portfolio_preflight.json',dict(status='PASS',profile=PROFILE,CPU_only=True,
        anchor_unchanged=True,nested_evidence_selection=True,score_forecast='NOT_AVAILABLE'))


--- CODE CELL 52 ---
# EXECUTE_PIPELINE
if __name__ == '__main__':
    RESULT=run_pipeline(CFG)
    print(RESULT['submission'].head().to_string(index=False))
    print('SUBMIT ONLY:', RESULT['submission_path'])
    print('DIAGNOSTICS:', RESULT['diagnostics_archive'])
    try:
        from IPython.display import display, FileLink
        display(FileLink(str(RESULT['submission_path'])))
    except ImportError:
        pass


