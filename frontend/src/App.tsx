import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Cpu, Radar, Activity, Power, RefreshCw } from 'lucide-react';
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
    onSuccess: () => alert("Trade Command Sent"),
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
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-6 max-w-[1800px] mx-auto pb-20">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-6 pb-6 border-b border-white/5 gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-3">
            <Activity className="text-accent w-8 h-8" /> TITANIUM <span className="text-accent neon-text">v18.6</span>
          </h1>
          <p className="text-secondary text-xs uppercase tracking-[0.3em] mt-1 opacity-60">Quant-First Asset Management</p>
        </div>
        
        {/* STATUS INDICATOR */}
        <div className={`px-6 py-2 rounded-full font-mono text-sm font-bold border flex items-center gap-2 ${s.is_active ? 'bg-accent/10 border-accent text-accent animate-pulse' : 'bg-danger/10 border-danger text-danger'}`}>
          <div className={`w-2 h-2 rounded-full ${s.is_active ? 'bg-accent' : 'bg-danger'}`}></div>
          {s.is_active ? 'ENGINE ONLINE' : 'ENGINE OFFLINE'}
        </div>
      </header>

      {/* === MASTER CONTROL BAR (ALWAYS VISIBLE) === */}
      <div className="glass-panel p-4 mb-8 flex flex-col md:flex-row gap-4 items-center justify-between border-white/10">
        <div className="flex gap-4 w-full md:w-auto">
          {s.is_active ? (
            <button onClick={() => stop.mutate()} className="flex-1 md:flex-none bg-danger/20 hover:bg-danger text-danger hover:text-white border border-danger px-8 py-3 rounded-xl font-bold transition flex items-center justify-center gap-2">
              <Power size={18} /> STOP ENGINE
            </button>
          ) : (
            <button onClick={() => start.mutate()} className="flex-1 md:flex-none bg-accent hover:bg-emerald-400 text-black px-8 py-3 rounded-xl font-bold transition flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(16,185,129,0.4)]">
              <Zap size={18} /> START ENGINE
            </button>
          )}
          
          <button 
            onClick={() => scan.mutate()} 
            disabled={scan.isPending}
            className="flex-1 md:flex-none bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-xl font-bold transition flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Radar size={18} className={scan.isPending ? 'animate-spin' : ''} />
            {scan.isPending ? 'SCANNING...' : 'SCAN MARKET'}
          </button>
        </div>

        <div className="flex gap-2 items-center bg-black/30 p-2 rounded-xl border border-white/5 w-full md:w-auto">
            <span className="text-[10px] text-secondary font-bold px-2">MANUAL</span>
            <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-16 bg-transparent text-center font-bold text-white border-b border-white/20 focus:border-accent outline-none" />
            <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="bg-accent/20 text-accent hover:bg-accent hover:text-black px-4 py-1.5 rounded-lg font-bold text-xs transition">BUY</button>
            <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="bg-danger/20 text-danger hover:bg-danger hover:text-white px-4 py-1.5 rounded-lg font-bold text-xs transition">SELL</button>
        </div>
      </div>

      {/* NAV TABS */}
      <nav className="flex gap-2 mb-8 overflow-x-auto pb-2 no-scrollbar">
        {[
          {id: 'live', label: 'DASHBOARD', icon: LayoutDashboard},
          {id: 'sim', label: 'BACKTEST', icon: Play},
          {id: 'health', label: 'SYSTEM', icon: ShieldCheck},
          {id: 'logs', label: 'LOGS', icon: Terminal}
        ].map(t => (
          <button 
            key={t.id} 
            onClick={() => setTab(t.id)} 
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold text-xs tracking-wider transition-all border ${tab === t.id ? 'bg-white/10 border-white/20 text-white' : 'bg-transparent border-transparent text-secondary hover:text-white'}`}
          >
            <t.icon size={16} className={tab === t.id ? 'text-accent' : ''} /> {t.label}
          </button>
        ))}
      </nav>

      {/* === DASHBOARD TAB === */}
      {tab === 'live' && (
        <div className="space-y-6 animate-in fade-in">
          {/* METRICS */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
             <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Total Equity</p><p className="text-2xl md:text-3xl font-bold mt-1 text-white neon-text">${s.equity?.toLocaleString() || '0.00'}</p></Card>
             <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Daily PnL</p><p className={`text-2xl md:text-3xl font-bold mt-1 ${s.daily_pnl>=0?'text-accent':'text-danger'}`}>{s.daily_pnl>=0?'+':''}${s.daily_pnl?.toLocaleString()}</p></Card>
             <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Regime</p><p className="text-2xl md:text-3xl font-bold mt-1 text-white">{s.regime || 'WAITING'}</p></Card>
             <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Quality</p><p className="text-2xl md:text-3xl font-bold mt-1 text-blue-400">{sig.quality ? sig.quality.toFixed(0) : 0}/100</p></Card>
             <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">API Usage</p><p className="text-2xl md:text-3xl font-bold mt-1 text-orange-400">{s.api_usage || 0}</p></Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <EquityChart data={d.history || []} />
              <MarketChart data={d.history || []} />
            </div>
            
            <div className="space-y-6">
              <SignalCard signal={sig} />
              <div className="glass-panel h-[300px] flex flex-col">
                <div className="p-4 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary tracking-widest flex justify-between items-center">
                    <span>LIVE LOGS</span>
                    <RefreshCw size={12} className="opacity-50"/>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-2 opacity-80">
                  {logData.slice().reverse().map((l: any, i: number) => (
                    <div key={i} className="leading-relaxed break-words">
                      <span className={l.level==='ERROR'?'text-danger':'text-accent'}>[{l.level}]</span> <span className="text-gray-400">{l.message}</span>
                    </div>
                  ))}
                </div>
              </div>
              <NewsFeed />
            </div>
          </div>
          <TradeTable trades={d.trades || []} />
        </div>
      )}

      {/* === BACKTEST TAB === */}
      {tab === 'sim' && (
        <div className="glass-panel p-10 text-center animate-in fade-in">
          <h2 className="text-2xl font-bold mb-4">Walk-Forward Simulation</h2>
          <button onClick={() => backtest.mutate()} className="bg-accent text-black px-8 py-3 rounded font-bold mb-8 hover:scale-105 transition">{backtest.isPending ? 'COMPUTING...' : 'RUN SIMULATION'}</button>
          {btData && (
            <div className="h-80 w-full bg-black/20 rounded-xl p-4 border border-white/5">
              <ResponsiveContainer><AreaChart data={btData.equity_curve}><YAxis hide/><Tooltip contentStyle={{background:'#000'}}/ ><Area dataKey="value" stroke="#10B981" fill="#10B98122"/></AreaChart></ResponsiveContainer>
              <div className="grid grid-cols-4 gap-4 mt-6">
                <div><p className="text-xs text-secondary">Return</p><p className="font-bold text-xl text-accent">{(btData.stats['Total Return']*100).toFixed(2)}%</p></div>
                <div><p className="text-xs text-secondary">Sharpe</p><p className="font-bold text-xl">{btData.stats['Sharpe']?.toFixed(2)}</p></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* === HEALTH TAB === */}
      {tab === 'health' && (
        <div className="grid gap-4 animate-in fade-in">
          <button onClick={() => diag.refetch()} className="bg-white/10 text-white py-4 rounded font-bold hover:bg-white/20">RUN SYSTEM DIAGNOSTICS</button>
          {diag.data?.map((c:any, i:number) => (
             <div key={i} className="glass-panel p-4 flex justify-between items-center"><span className="font-bold">{c.name}</span><span className={`px-3 py-1 rounded text-xs font-bold ${c.status==='PASS'?'bg-accent/20 text-accent':'bg-danger/20 text-danger'}`}>{c.status}</span></div>
          ))}
        </div>
      )}

      {/* === LOGS TAB === */}
      {tab === 'logs' && (
        <div className="glass-panel h-[800px] overflow-y-auto p-4 font-mono text-xs space-y-1 animate-in fade-in">
          {logData.map((l:any, i:number) => <div key={i} className="border-b border-white/5 pb-1"><span className="text-secondary opacity-50 mr-2">{l.timestamp}</span> <span className={l.level==='ERROR'?'text-danger':'text-accent'}>{l.level}</span> {l.message}</div>)}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}
