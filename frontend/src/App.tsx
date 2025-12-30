import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Cpu, Radar, Activity } from 'lucide-react';
import { EquityChart } from './components/dashboard/EquityChart';
import { MarketChart } from './components/dashboard/MarketChart';
import { SignalCard } from './components/dashboard/SignalCard';
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
    onSuccess: () => alert("Trade Sent!"),
    onError: (e: any) => alert("Error: " + (e.response?.data?.detail || e.message))
  });
  const scan = useMutation({ mutationFn: () => api.post('/tools/scan'), onSuccess: () => state.refetch() });
  
  return { state, logs, backtest, diag, start, stop, trade, scan };
}

const Card = ({ children, className }: any) => <div className={`glass-panel p-6 ${className}`}>{children}</div>;

export default function App() {
  const { state, logs, backtest, diag, start, stop, trade, scan } = useTitanium();
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
      {/* HEADER */}
      <header className="flex justify-between items-center mb-6 pb-6 border-b border-white/5">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-3">
            <Activity className="text-accent" /> TITANIUM <span className="text-accent neon-text">v18.6</span>
          </h1>
        </div>
        
        {/* STATUS BADGE */}
        <div className={`px-6 py-2 rounded-full font-mono text-sm font-bold border ${s.is_active ? 'text-accent border-accent/40 animate-pulse' : 'text-danger border-danger/40'}`}>
          {s.is_active ? '● ONLINE' : '● OFFLINE'}
        </div>
      </header>

      {/* EMERGENCY CONTROL BAR - ALWAYS VISIBLE */}
      <div className="flex gap-4 mb-8 bg-surface p-4 rounded-xl border border-white/10">
        <button 
          onClick={() => start.mutate()} 
          className="bg-success text-black px-6 py-2 rounded font-bold hover:brightness-110 flex items-center gap-2"
        >
          <Play size={16}/> RESUME ENGINE
        </button>
        
        <button 
          onClick={() => scan.mutate()} 
          disabled={scan.isPending}
          className="bg-blue-600 text-white px-6 py-2 rounded font-bold hover:brightness-110 flex items-center gap-2"
        >
          <Radar size={16} className={scan.isPending ? 'animate-spin' : ''}/> 
          {scan.isPending ? 'SCANNING...' : 'SCAN MARKET'}
        </button>

        <button 
          onClick={() => stop.mutate()} 
          className="bg-danger/20 text-danger border border-danger px-6 py-2 rounded font-bold hover:bg-danger hover:text-white transition ml-auto"
        >
          HALT SYSTEM
        </button>
      </div>

      {/* NAV */}
      <nav className="flex gap-4 mb-8 border-b border-white/5 pb-1 overflow-x-auto">
        {[{id: 'live', label: 'COMMAND'}, {id: 'sim', label: 'BACKTEST'}, {id: 'health', label: 'SYSTEM'}, {id: 'logs', label: 'LOGS'}].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`pb-4 px-4 font-bold text-sm border-b-2 transition ${tab === t.id ? 'border-accent text-accent' : 'border-transparent text-secondary'}`}>
            {t.label}
          </button>
        ))}
      </nav>

      {/* LIVE TAB */}
      {tab === 'live' && (
        <div className="space-y-6 animate-in fade-in">
          {/* METRICS */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
             <Card>
               <p className="text-xs text-secondary font-bold">TOTAL EQUITY</p>
               <p className="text-2xl font-bold text-white mt-1">${s.equity?.toLocaleString() || '0.00'}</p>
             </Card>
             <Card>
               <p className="text-xs text-secondary font-bold">REGIME</p>
               <p className="text-2xl font-bold text-accent mt-1">{s.regime || 'WAITING'}</p>
             </Card>
             <Card>
               <p className="text-xs text-secondary font-bold">SIGNAL</p>
               <p className={`text-2xl font-bold mt-1 ${sig.sentiment==='BULLISH'?'text-success':'text-danger'}`}>{sig.sentiment || 'NONE'}</p>
             </Card>
             <Card>
               <p className="text-xs text-secondary font-bold">API USAGE</p>
               <p className="text-2xl font-bold text-orange-400 mt-1">{s.api_usage || 0}/800</p>
             </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <EquityChart data={d.history || []} />
              <MarketChart data={d.history || []} />
            </div>
            
            <div className="space-y-6">
              {/* MANUAL TRADE */}
              <Card>
                 <h3 className="font-bold mb-4 text-xs text-secondary">MANUAL OVERRIDE (GLD)</h3>
                 <div className="flex gap-2">
                   <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-20 bg-black/30 border border-white/10 rounded p-2 text-center text-white" />
                   <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 bg-accent/20 text-accent border border-accent/50 rounded font-bold text-xs hover:bg-accent hover:text-black transition">BUY</button>
                   <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 bg-danger/20 text-danger border border-danger/50 rounded font-bold text-xs hover:bg-danger hover:text-white transition">SELL</button>
                 </div>
              </Card>

              <SignalCard signal={sig} />
              
              <div className="glass-panel h-[300px] flex flex-col">
                <div className="p-3 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary">LIVE LOGS</div>
                <div className="flex-1 overflow-y-auto p-3 font-mono text-[10px] space-y-2 opacity-80">
                  {logData.slice().reverse().map((l: any, i: number) => (
                    <div key={i}><span className={l.level==='ERROR'?'text-danger':'text-accent'}>[{l.level}]</span> {l.message}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <TradeTable trades={d.trades || []} />
        </div>
      )}

      {/* BACKTEST TAB */}
      {tab === 'sim' && (
        <div className="glass-panel p-10 text-center animate-in fade-in">
          <button onClick={() => backtest.mutate()} className="bg-accent text-black px-8 py-3 rounded font-bold mb-8">RUN SIMULATION</button>
          {btData && (
            <div className="h-80 w-full bg-black/20 rounded-xl p-4">
              <ResponsiveContainer><AreaChart data={btData.equity_curve}><YAxis hide/><Tooltip contentStyle={{background:'#000'}}/ ><Area dataKey="value" stroke="#10B981" fill="#10B98122"/></AreaChart></ResponsiveContainer>
              <div className="grid grid-cols-4 gap-4 mt-4">
                <div><p className="text-xs text-secondary">Return</p><p className="font-bold text-xl text-accent">{(btData.stats['Total Return']*100).toFixed(2)}%</p></div>
                <div><p className="text-xs text-secondary">Sharpe</p><p className="font-bold text-xl">{btData.stats['Sharpe']?.toFixed(2)}</p></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* HEALTH TAB */}
      {tab === 'health' && (
        <div className="grid gap-4 animate-in fade-in">
          <button onClick={() => diag.refetch()} className="bg-white/10 text-white py-3 rounded font-bold">SCAN SYSTEM HEALTH</button>
          {diag.data?.map((c:any, i:number) => (
             <div key={i} className="glass-panel p-4 flex justify-between"><span className="font-bold">{c.name}</span><span className={c.status==='PASS'?'text-accent':'text-danger'}>{c.status}</span></div>
          ))}
        </div>
      )}

      {/* LOGS TAB */}
      {tab === 'logs' && (
        <div className="glass-panel h-[800px] overflow-y-auto p-4 font-mono text-xs space-y-1 animate-in fade-in">
          {logData.map((l:any, i:number) => <div key={i}><span className="text-secondary">{l.timestamp}</span> <span className={l.level==='ERROR'?'text-danger':'text-accent'}>{l.level}</span> {l.message}</div>)}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}
