from __future__ import annotations
import argparse, json, math, platform, time
import numpy as np
import torch
from poc import ROOT,GPT,GPTConfig,seed_all,dump_json,make_synthetic_data,teacher_targets

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1600)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args();seed_all(args.seed,args.threads)
    result=ROOT/'results';result.mkdir(exist_ok=True)
    meta=make_synthetic_data(ROOT/'data')
    data={name:torch.tensor(np.load(ROOT/'data'/f'{name}.npy').astype(np.int64)) for name in ['train','dev','test']}
    config=GPTConfig(block_size=64,vocab_size=meta['vocab_size'],n_layer=2,n_head=4,n_embd=64,dropout=0.0,bias=True)
    model=GPT(config)
    optim=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01)
    batch_size=32;window=config.block_size
    rng=torch.Generator().manual_seed(args.seed+500)
    def batch(split,bs=batch_size):
        seq=data[split];idx=torch.randint(len(seq)-window-1,(bs,),generator=rng)
        xx=seq[idx[:,None]+torch.arange(window)[None,:]]
        yy=seq[idx[:,None]+1+torch.arange(window)[None,:]]
        return xx,yy
    @torch.no_grad()
    def evaluate(split):
        model.eval();losses=[]
        erng=torch.Generator().manual_seed(123 if split=='dev' else 124)
        seq=data[split]
        for _ in range(12):
            ix=torch.randint(len(seq)-window-1,(32,),generator=erng)
            xx=seq[ix[:,None]+torch.arange(window)[None,:]]
            yy=seq[ix[:,None]+1+torch.arange(window)[None,:]]
            losses.append(model(xx,yy)[1].item())
        return float(np.mean(losses))
    env={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu',
         'cuda_available':torch.cuda.is_available(),'threads':args.threads,'dtype':'float32',
         'batch_size_train':batch_size,'batch_size_decode':1,'compile':False,'kv_cache':False,
         'lora_merged_at_benchmark':True,'clock':'not fixed','seed_teacher':args.seed,
         'config':vars(config),'num_parameters':sum(q.numel() for q in model.parameters())}
    dump_json(result/'environment.json',env)
    initial=evaluate('dev');print(json.dumps({'stage':'teacher','step':0,'dev_nll':initial}),flush=True)
    start=time.perf_counter();best=float('inf');history=[]
    for step in range(1,args.steps+1):
        model.train();lr=0.003*(0.15+0.85*0.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optim.param_groups:group['lr']=lr
        x,y=batch('train');optim.zero_grad(set_to_none=True)
        _,loss=model(x,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        if step%200==0 or step==args.steps:
            val=evaluate('dev');row={'stage':'teacher','step':step,'train_nll':loss.item(),'dev_nll':val,'elapsed_s':time.perf_counter()-start}
            history.append(row);print(json.dumps(row),flush=True)
            if val<best:
                best=val;torch.save({'model':model.state_dict(),'config':vars(config),'step':step,'seed':args.seed},result/'teacher.pt')
    saved=torch.load(result/'teacher.pt',weights_only=True,map_location='cpu');model.load_state_dict(saved['model']);model.eval()
    dev=evaluate('dev');test=evaluate('test')
    chars=meta['chars'];prefix_text='record 20500: alice found a '
    prefix=torch.tensor([[chars.index(c) for c in prefix_text]])
    sample=teacher_targets(model,prefix,192)[0].tolist()
    resultdata={'initial_dev_nll':initial,'best_dev_nll':dev,'test_nll':test,'steps':args.steps,
                'seconds':time.perf_counter()-start,'history':history,
                'greedy_sample':prefix_text+''.join(chars[t] for t in sample)}
    dump_json(result/'teacher_training.json',resultdata)
    print(json.dumps({'stage':'teacher_done','dev_nll':dev,'test_nll':test,'seconds':resultdata['seconds']}),flush=True)
if __name__=='__main__':main()
