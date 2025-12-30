import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Cpu, Radar, Database, Activity } from 'lucide-react';
import { EquityChart } from './components/dashboard/EquityChart';
import { MarketChart } from './components/dashboard/MarketChart';
import { NewsFeed } from './components/dashboard/NewsFeed';
import { TradeTable } from './components/dashboard/TradeTable';

function useTitanium() {
  const state = useQuery({ queryKey: ['state'], queryFn: () => api.get('/status').then(r => r.data), refetchInterval: 2000 });
  const logs = useQuery({ queryKey: ['logs'], queryFn: () => api.get('/logs?limit=50').then(r => r.data), refetchInterval: 3000 });
  const backtest = useMutation({ mutationFn: () => api.post('/tools/backtest?days=180') });
  const diag = useQuery({ queryKey: ['diag'], queryFn: () => api.get('/tools/diagnostics').then(r => r.data), enabled: false });
  const start = useMutation({ mutationFn: () => api.post('/control/start') });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop') });
  const trade = useMutation({ 
    mutationFn: (p: any) => api.post('/trade/force', p),
    onSuccess: () => alert("Trade Sent to Engine"),
    onError: (e: any) => alert("Trade Failed: " + (e.response?.data?.detail || e.message))
  });
  
  return { state, logs, backtest, diag, start, stop, trade };
}

const Card = ({ children, className }: any) => <div className={`glass-panel p-6 ${className}`}>{children}</div>;

export default function App() {
  const { state, logs, backtest, diag, start, stop, trade } = useTitanium();
  const [tab, setTab] = useState('live');
  const [qty, setQty] = useState(10);
  const logEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs.data]);

  const d = state.data || {};
  const s = d.state || {};
  const sig = d.signal || {};
  const logData = logs.data || [];
  const btData = backtest.data?.data;

  return (
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-8 max-w-[1700px] mx-auto">
      <header className="flex justify-between items-center mb-10 border-b border-white/5 pb-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3 drop-shadow-lg">
            <Activity className="text-accent w-10 h-10" /> TITANIUM <span className="text-accent neon-text">v18.6</span>
          </h1>
          <p className="text-secondary text-xs uppercase tracking-[0.3em] mt-2 opacity-60">HMM Quantitative System</p>
        </div>
        <div className={`px-8 py-3 rounded-2xl glass-card font-mono text-sm font-bold border ${s.is_active ? 'text-accent border-accent/40 animate-pulse' : 'text-danger border-danger/40'}`}>
          {s.is_active ? '● ENGINE ACTIVE' : '● ENGINE OFFLINE'}
        </div>
      </header>

      <nav className="flex gap-4 mb-10 border-b border-white/5 pb-1 overflow-x-auto">
        {[{id: 'live', label: 'COMMAND', icon: LayoutDashboard}, {id: 'sim', label: 'BACKTEST', icon: Play}, {id: 'health', label: 'SYSTEM', icon: ShieldCheck}, {id: 'logs', label: 'LOGS', icon: Terminal}].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`pb-4 px-4 flex items-center gap-2 text-xs font-bold tracking-widest transition-all border-b-2 whitespace-nowrap hover:text-white ${tab === t.id ? 'border-accent text-accent text-glow' : 'border-transparent text-secondary'}`}>
            <t.icon size={14} /> {t.label}
          </button>
        ))}
      </nav>

      {tab === 'live' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-700">
          
          {/* V18.6 METRICS */}
          <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-5 gap-6">
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Total Equity</p><p className="text-3xl font-bold mt-2 text-white neon-text">${s.equity?.toLocaleString()}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Daily PnL</p><p className={`text-3xl font-bold mt-2 ${s.daily_pnl>=0?'text-accent':'text-danger'}`}>{s.daily_pnl>=0?'+':''}${s.daily_pnl?.toLocaleString()}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Active Regime</p><p className="text-3xl font-bold mt-2 text-white">{s.regime || 'WAITING'}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Signal Quality</p><p className="text-3xl font-bold mt-2 text-blue-400">{sig.quality ? sig.quality.toFixed(1) : 0}/100</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">API Budget</p><p className="text-3xl font-bold mt-2 text-orange-400">{s.api_usage || 0}/800</p></Card>
          </div>

          <div className="lg:col-span-8 space-y-8">
            <EquityChart data={d.history || []} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card>
                <h3 className="text-xs font-bold text-secondary uppercase tracking-widest mb-6 flex items-center gap-2"><Cpu size={14}/> Engine Control</h3>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <button onClick={() => start.mutate()} className="btn-cyber text-accent">RESUME</button>
                  <button onClick={() => stop.mutate()} className="btn-cyber-danger text-danger">HALT</button>
                </div>
                <div className="flex gap-2 p-1 bg-black/40 rounded-xl border border-white/5">
                  <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="bg-transparent w-16 text-center text-white font-bold outline-none text-sm" />
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 text-accent font-bold text-[10px] hover:bg-white/5 rounded transition">BUY</button>
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 text-danger font-bold text-[10px] hover:bg-white/5 rounded transition">SELL</button>
                </div>
              </Card>
              <NewsFeed />
            </div>
          </div>

          <div className="lg:col-span-4 space-y-8">
            {/* SIGNAL STATUS */}
            <Card className="h-64 flex flex-col justify-center items-center text-center">
               <Radar size={48} className={`mb-4 ${sig.sentiment==='BULLISH'?'text-accent':'text-danger'}`} />
               <h2 className="text-2xl font-bold text-white">{sig.sentiment || 'SCANNING'}</h2>
               <p className="text-secondary text-xs mt-2">Score: {sig.score?.toFixed(3) || 0} | Timeframe: {sig.timeframe || '1d'}</p>
            </Card>

            <div className="glass-panel h-[400px] flex flex-col">
              <div className="p-4 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary tracking-widest">LIVE LOGS</div>
              <div className="flex-1 overflow-y-auto p-5 font-mono text-[10px] space-y-3 opacity-80">
                {logData.slice().reverse().map((l: any, i: number) => (
                  <div key={i} className="flex gap-3"><span className="text-gray-600 shrink-0">{l.timestamp?.split('T')[1]?.split('.')[0]}</span><span className={l.level==='ERROR'?'text-danger':'text-accent'}>[{l.level}]</span><span className="text-gray-300">{l.message}</span></div>
                ))}
              </div>
            </div>
          </div>
          <div className="lg:col-span-12"><TradeTable trades={d.trades || []} /></div>
        </div>
      )}

      {/* SIM TAB */}
      {tab === 'sim' && (
        <div className="glass-panel p-10 text-center animate-in fade-in">
          <h2 className="text-2xl font-bold mb-4">Walk-Forward Analysis</h2>
          <button onClick={() => backtest.mutate()} className="btn-cyber text-accent">{backtest.isPending ? 'RUNNING SIMULATION...' : 'RUN V18.6 BACKTEST'}</button>
          
          {btData && (
            <div className="mt-10 space-y-8">
               <div className="grid grid-cols-4 gap-4">
                 <Card><p className="text-xs text-secondary">Total Return</p><p className="text-2xl font-bold text-accent">{(btData.stats['Total Return'] * 100).toFixed(2)}%</p></Card>
                 <Card><p className="text-xs text-secondary">Sharpe</p><p className="text-2xl font-bold">{btData.stats['Sharpe']?.toFixed(2)}</p></Card>
                 <Card><p className="text-xs text-secondary">Drawdown</p><p className="text-2xl font-bold text-danger">{(btData.stats['Max DD'] * 100).toFixed(2)}%</p></Card>
                 <Card><p className="text-xs text-secondary">Folds</p><p className="text-2xl font-bold">{btData.stats['Completed Folds']}</p></Card>
               </div>
               <div className="h-80 w-full bg-black/20 rounded-xl p-4">
                 <ResponsiveContainer><AreaChart data={btData.equity_curve}><YAxis domain={['auto', 'auto']} hide /><Tooltip contentStyle={{background:'#000', border:'1px solid #333'}}/><Area dataKey="value" stroke="#10B981" fill="#10B98122"/></AreaChart></ResponsiveContainer>
               </div>
            </div>
          )}
        </div>
      )}

      {/* HEALTH TAB */}
      {tab === 'health' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in">
           {diag.data?.map((c: any, i: number) => (
             <div key={i} className="glass-panel p-6 flex justify-between items-center"><span className="font-bold text-lg">{c.name}</span><span className={`px-4 py-1 rounded-full text-xs font-bold ${c.status==='PASS'?'bg-success/20 text-success':'bg-danger/20 text-danger'}`}>{c.status}</span></div>
           ))}
           <button onClick={() => diag.refetch()} className="col-span-full btn-cyber text-white">RE-SCAN SYSTEM</button>
        </div>
      )}

      {/* LOGS TAB */}
      {tab === 'logs' && (
        <div className="glass-panel h-[800px] flex flex-col animate-in fade-in">
           <div className="p-6 border-b border-white/5 font-bold tracking-widest text-secondary">FULL SYSTEM LOGS</div>
           <div className="flex-1 overflow-y-auto p-6 font-mono text-xs space-y-2">
             {logData.map((l:any, i:number) => <div key={i} className="border-b border-white/5 pb-1 opacity-70 hover:opacity-100"><span className={`mr-2 ${l.level==='ERROR'?'text-danger':l.level==='WARNING'?'text-warning':'text-accent'}`}>[{l.level}]</span>{l.message}</div>)}
             <div ref={logEndRef} />
           </div>
        </div>
      )}
    </div>
  );
}
