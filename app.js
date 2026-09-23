const $ = window.$ = (id) => document.getElementById(id);

$('ts').onclick = () => {
  $('ts').classList.add('active'); $('to').classList.remove('active');
  $('ps').classList.add('active'); $('po').classList.remove('active');
};
$('to').onclick = () => {
  $('to').classList.add('active'); $('ts').classList.remove('active');
  $('po').classList.add('active'); $('ps').classList.remove('active');
  setTimeout(() => window.dispatchEvent(new Event('resize')), 20);
};

(() => {
  const cv = $('sim');
  const ctx = cv.getContext('2d', {alpha:false});

  // Core model constants restored from the fuller Monte Carlo version.
  const N = 90, SZ = N * N;
  const T0 = 1540, Tliq0 = 1500, alpha = 0.17;
  const latentHeat = 7.0;
  const attemptsPerStep = Math.floor(0.75 * SZ);
  const nucleationPrefactor = 0.0018;
  const growthPrefactor = 0.40;
  const criticalUndercooling = 7.0;
  const growthScale = 18.0;

  const T  = new Float32Array(SZ);
  const G  = new Int32Array(SZ);
  const Sl = new Float32Array(SZ);
  const Pl = new Float32Array(SZ);
  const Ss = new Float32Array(SZ);
  const Ps = new Float32Array(SZ);
  const orient = [0];

  let nextG = 1, stepN = 0, running = false, raf = null, last = 0;

  const id = (x,y) => y*N+x;
  const P = () => ({
    s:+$('s').value, p:+$('p').value, D:+$('d').value, passes:+$('dp').value,
    ks:+$('ks').value, kp:+$('kp').value, wall:+$('wall').value,
    growth:+$('g').value, nuc:+$('n').value, speed:+$('speed').value
  });

  function nb4(k){
    const x=k%N, y=(k/N)|0, out=[];
    if(x>0) out.push(k-1); if(x<N-1) out.push(k+1);
    if(y>0) out.push(k-N); if(y<N-1) out.push(k+N);
    return out;
  }
  function nb8(k){
    const x=k%N, y=(k/N)|0, out=[];
    for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){
      if(!dx && !dy) continue;
      const X=x+dx, Y=y+dy;
      if(X>=0 && X<N && Y>=0 && Y<N) out.push(id(X,Y));
    }
    return out;
  }
  function solidNeighbors(k){
    const out=[]; for(const j of nb8(k)) if(G[j]) out.push(j); return out;
  }
  function distinctNeighborGrains(k){
    const s=new Set(); for(const j of nb8(k)) if(G[j]) s.add(G[j]); return s.size;
  }
  function channelScore(k){
    if(G[k]) return 0;
    const dg=distinctNeighborGrains(k);
    let sn=0; for(const j of nb8(k)) if(G[j]) sn++;
    return 2.2*Math.max(0,dg-1) + 0.18*sn;
  }
  function solidFraction(){
    let n=0; for(let k=0;k<SZ;k++) if(G[k]) n++;
    return n/SZ;
  }

  function reset(){
    running=false; if(raf) cancelAnimationFrame(raf);
    $('run').textContent='Run';
    const a=P();
    nextG=1; stepN=0; orient.length=1;
    for(let k=0;k<SZ;k++){
      T[k]=T0 + (Math.random()-.5)*0.5;
      G[k]=0; Sl[k]=a.s; Pl[k]=a.p; Ss[k]=0; Ps[k]=0;
    }
    draw(); stats(); $('status').textContent='Ready';
  }

  function diffuseTemperature(){
    const a=P(), old=T.slice();
    for(let y=1;y<N-1;y++) for(let x=1;x<N-1;x++){
      const k=id(x,y);
      T[k]=old[k]+alpha*(old[k-1]+old[k+1]+old[k-N]+old[k+N]-4*old[k]);
    }
    for(let x=0;x<N;x++){ T[x]=a.wall; T[(N-1)*N+x]=a.wall; }
    for(let y=0;y<N;y++){ T[y*N]=a.wall; T[y*N+N-1]=a.wall; }
  }

  // Conservative pairwise diffusion: every liquid-liquid flux is added to one
  // cell and subtracted from the other, so solute is not numerically created.
  function diffuseLiquid(field, strength){
    const old=field.slice(), delta=new Float32Array(SZ);
    const f=Math.min(0.10, Math.max(0, strength*0.22));
    const drift=Math.min(0.045, Math.max(0.008, strength*0.08));
    for(let y=1;y<N-1;y++) for(let x=1;x<N-1;x++){
      const k=id(x,y); if(G[k]) continue;
      for(const j of [k+1,k+N]){
        if(G[j]) continue;
        const flux=f*(old[j]-old[k]);
        delta[k]+=flux; delta[j]-=flux;

        // Weak conservative segregation drift toward more constrained
        // interdendritic liquid (multi-grain channels / triple junctions).
        const sk=channelScore(k), sj=channelScore(j), ds=sj-sk;
        if(Math.abs(ds)>0.05){
          const donor=ds>0?k:j, recv=ds>0?j:k;
          const amount=Math.min(old[donor]*0.025, drift*Math.abs(ds)*old[donor]/8);
          delta[donor]-=amount; delta[recv]+=amount;
        }
      }
    }
    for(let k=0;k<SZ;k++) if(!G[k]) field[k]=Math.max(0, old[k]+delta[k]);
  }
  function diffuseSolute(){
    const a=P();
    for(let r=0;r<a.passes;r++){ diffuseLiquid(Sl,a.D); diffuseLiquid(Pl,a.D); }
  }

  function nearestLiquidRecipients(k){
    let frontier=[k], seen=new Uint8Array(SZ); seen[k]=1;
    for(let radius=1; radius<=5; radius++){
      const next=[], found=[];
      for(const q of frontier){
        for(const j of nb8(q)){
          if(seen[j]) continue; seen[j]=1;
          if(!G[j]) found.push(j); else next.push(j);
        }
      }
      if(found.length) return found;
      frontier=next;
      if(!frontier.length) break;
    }
    return [];
  }

  function chooseGrowthGrain(k, neigh){
    const x=k%N, y=(k/N)|0, weights=[], gids=[];
    let total=0;
    for(const j of neigh){
      const gid=G[j], jx=j%N, jy=(j/N)|0;
      const theta=Math.atan2(y-jy,x-jx);
      const o=orient[gid] || 0;
      const anis=0.72 + 0.28*Math.abs(Math.cos(4*(theta-o)));
      const w=anis*(1 + 0.12*neigh.filter(q=>G[q]===gid).length);
      gids.push(gid); weights.push(w); total+=w;
    }
    let r=Math.random()*total;
    for(let i=0;i<weights.length;i++){ r-=weights[i]; if(r<=0) return gids[i]; }
    return gids[gids.length-1];
  }

  function freeze(k, gid){
    const a=P(), cS=Sl[k], cP=Pl[k];
    const fs=solidFraction();
    const ch=channelScore(k);
    // Scheil-like rejection early, then increasing terminal trapping in the
    // last interdendritic liquid so the final boundary network retains solute.
    const trap=Math.max(0,Math.min(0.92,(fs-0.82)/0.18))*Math.min(1,ch/3.2);
    const kS=a.ks+(1-a.ks)*trap;
    const kP=a.kp+(1-a.kp)*trap;
    Ss[k]=kS*cS; Ps[k]=kP*cP;
    const rejectS=(1-kS)*cS, rejectP=(1-kP)*cP;
    G[k]=gid; Sl[k]=0; Pl[k]=0;

    let rec=nb8(k).filter(j=>!G[j]);
    if(!rec.length) rec=nearestLiquidRecipients(k);

    if(rec.length){
      const w=[]; let sum=0;
      for(const j of rec){
        // Prefer liquid cells already constrained by multiple grains:
        // those are the interdendritic / grain-boundary channels that freeze last.
        const dg=Math.max(0,distinctNeighborGrains(j)-1);
        const q=1 + 7.0*dg*dg + 0.45*solidNeighbors(j).length + 1.8*channelScore(j);
        w.push(q); sum+=q;
      }
      rec.forEach((j,i)=>{ Sl[j]+=rejectS*w[i]/sum; Pl[j]+=rejectP*w[i]/sum; });
    } else {
      // Only true for the final isolated liquid: its remaining solute is trapped
      // when it finally solidifies.
      Ss[k]+=rejectS; Ps[k]+=rejectP;
    }
    T[k]+=latentHeat;
  }

  function localLiquidus(k){
    const a=P();
    const sr=Math.max(1, Sl[k]/Math.max(1,a.s));
    const pr=Math.max(1, Pl[k]/Math.max(1,a.p));
    return Tliq0 - 5.2*(sr-1) - 3.4*(pr-1);
  }

  function mc(){
    diffuseTemperature();
    diffuseSolute();
    const a=P();

    for(let attempt=0; attempt<attemptsPerStep; attempt++){
      const x=1+(Math.random()*(N-2)|0);
      const y=1+(Math.random()*(N-2)|0);
      const k=id(x,y);
      if(G[k]) continue;

      const undercool=localLiquidus(k)-T[k];
      if(undercool<=0) continue;

      const neigh=solidNeighbors(k);
      if(neigh.length){
        const enrichment=Sl[k]/Math.max(1,a.s);
        const soluteDrag=1/(1+0.035*Math.max(0,enrichment-1));
        const dg=distinctNeighborGrains(k);
        const boundarySlow=dg>=3 ? 0.08 : (dg>=2 ? 0.22 : 1.0);
        const pGrow=Math.min(
          0.98,
          growthPrefactor*a.growth*soluteDrag*boundarySlow*(1-Math.exp(-undercool/growthScale))
        );
        if(Math.random()<pGrow){
          freeze(k, chooseGrowthGrain(k,neigh));
        }
      } else if(undercool>criticalUndercooling){
        const z=(undercool-criticalUndercooling)/20;
        const pNuc=Math.min(0.15, nucleationPrefactor*a.nuc*z*z);
        if(Math.random()<pNuc){
          orient[nextG]=Math.random()*Math.PI/2;
          freeze(k,nextG++);
        }
      }
    }

    diffuseSolute();
    stepN++;
  }

  const palette=[
    [57,106,177],[218,124,48],[62,150,81],[204,37,41],[107,76,154],[146,36,40],
    [83,81,84],[148,139,61],[0,154,205],[255,128,14],[95,158,209],[237,151,202]
  ];
  function cmap(t){
    t=Math.max(0,Math.min(1,t));
    const stops=[[16,30,92],[20,82,145],[23,150,168],[89,191,111],[218,207,65],[220,103,38],[190,47,30]];
    const q=t*(stops.length-1), i=Math.min(stops.length-2,Math.floor(q)), f=q-i;
    const A=stops[i], B=stops[i+1];
    return [A[0]+(B[0]-A[0])*f,A[1]+(B[1]-A[1])*f,A[2]+(B[2]-A[2])*f];
  }
  function ratio(k,type){
    const a=P();
    return type==='sulfur'
      ? (G[k]?Ss[k]/Math.max(1,a.s):Sl[k]/Math.max(1,a.s))
      : (G[k]?Ps[k]/Math.max(1,a.p):Pl[k]/Math.max(1,a.p));
  }

  function draw(){
    const mode=$('view').value, img=ctx.createImageData(N,N), d=img.data;
    for(let k=0;k<SZ;k++){
      let c;
      if(mode==='grains') c=G[k]?palette[(G[k]-1)%palette.length]:[15,20,27];
      else if(mode==='phase') c=G[k]?[214,214,218]:[17,24,32];
      else if(mode==='temperature') c=cmap((T[k]-1240)/340);
      else {
        const r=ratio(k,mode);
        c=cmap(Math.log1p(Math.max(0,r-0.85))/Math.log(11.0));
      }
      d[k*4]=c[0]; d[k*4+1]=c[1]; d[k*4+2]=c[2]; d[k*4+3]=255;
    }
    const off=document.createElement('canvas');
    off.width=N; off.height=N;
    off.getContext('2d').putImageData(img,0,0);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(off,0,0,cv.width,cv.height);
    $('legend').textContent={
      grains:'Each color is one grain. Growth includes a weak cubic orientation anisotropy.',
      sulfur:'Sulfur concentration. Dark grain interiors contrast with enriched connected late-liquid channels and junctions.',
      phosphorus:'Phosphorus concentration using the same connected-liquid partition/diffusion model.',
      temperature:'Temperature field with wall cooling, thermal diffusion and latent-heat feedback.',
      phase:'Light = solid; dark = liquid.'
    }[mode];
  }

  function stats(){
    const a=P(); let solid=0,sumT=0,peakS=0;
    for(let k=0;k<SZ;k++){
      if(G[k]) solid++;
      sumT+=T[k];
      peakS=Math.max(peakS,G[k]?Ss[k]/Math.max(1,a.s):Sl[k]/Math.max(1,a.s));
    }
    $('solid').textContent=(100*solid/SZ).toFixed(1)+'%';
    $('grains').textContent=nextG-1;
    $('temp').textContent=(sumT/SZ).toFixed(0)+' °C';
    $('peak').textContent=peakS.toFixed(1)+'×';
    $('st').textContent=stepN;
    $('status').textContent=running?'Running':solid/SZ>.995?'Essentially solid':'Paused';
  }

  function loop(ts){
    if(!running) return;
    if(ts-last>32){
      const n=P().speed;
      for(let i=0;i<n;i++) mc();
      draw(); stats(); last=ts;
    }
    raf=requestAnimationFrame(loop);
  }

  $('run').onclick=()=>{
    running=!running; $('run').textContent=running?'Pause':'Run';
    if(running){ last=0; raf=requestAnimationFrame(loop); }
    else if(raf) cancelAnimationFrame(raf);
  };
  $('step').onclick=()=>{ if(!running){ mc(); draw(); stats(); } };
  $('reset').onclick=reset;
  $('view').onchange=draw;

  [
    ['speed','speedV',v=>v+'×'],['s','sV',v=>v+' ppm'],['p','pV',v=>v+' ppm'],
    ['d','dV',v=>(+v).toFixed(2)],['dp','dpV',v=>v],['ks','ksV',v=>(+v).toFixed(2)],
    ['kp','kpV',v=>(+v).toFixed(2)],['wall','wallV',v=>v+' °C'],
    ['g','gV',v=>(+v).toFixed(1)+'×'],['n','nV',v=>(+v).toFixed(1)+'×']
  ].forEach(([a,b,f])=>{
    $(a).oninput=()=>$(b).textContent=f($(a).value);
    if(a!=='speed') $(a).onchange=reset;
  });

  reset();
})();