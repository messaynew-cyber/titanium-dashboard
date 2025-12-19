import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Cpu, Radar } from 'lucide-react';
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
    onSuccess: () => alert("Trade Executed Successfully"),
    onError: (err: any) => alert(`Trade Failed: ${err.response?.data?.detail || err.message}`)
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

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.data]);

  const d = state.data || {};
  const s = d.state || {};
  const logData = logs.data || [];

  return (
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-8 max-w-[1700px] mx-auto">
      {/* HEADER */}
      <header className="flex justify-between items-center mb-10 border-b border-white/5 pb-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3 drop-shadow-lg">
            <TrendingUp className="text-accent w-10 h-10" /> TITANIUM <span className="text-accent neon-text">X</span>
          </h1>
          <p className="text-secondary text-xs uppercase tracking-[0.3em] mt-2 opacity-60">Quant-First Asset Management</p>
        </div>
        <div className={`px-8 py-3 rounded-2xl glass-card font-mono text-sm font-bold border ${s.is_active ? 'text-accent border-accent/40 animate-pulse' : 'text-danger border-danger/40'}`}>
          {s.is_active ? '● ENGINE ACTIVE' : '● ENGINE OFFLINE'}
        </div>
      </header>

      {/* NAV */}
      <nav className="flex gap-4 mb-10 border-b border-white/5 pb-1 overflow-x-auto">
        {[{id: 'live', label: 'LIVE TERMINAL', icon: LayoutDashboard}, {id: 'sim', label: 'BACKTEST LAB', icon: Play}, {id: 'health', label: 'DIAGNOSTICS', icon: ShieldCheck}, {id: 'logs', label: 'CONSOLE', icon: Terminal}].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`pb-4 px-4 flex items-center gap-2 text-xs font-bold tracking-widest transition-all border-b-2 whitespace-nowrap hover:text-white ${tab === t.id ? 'border-accent text-accent text-glow' : 'border-transparent text-secondary'}`}>
            <t.icon size={14} /> {t.label}
          </button>
        ))}
      </nav>

      {/* LIVE TERMINAL */}
      {tab === 'live' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-700">
          {/* KPI */}
          <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Equity Value</p><p className="text-4xl font-bold mt-2 text-white neon-text">${s.equity?.toLocaleString()}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Daily PnL</p><p className={`text-4xl font-bold mt-2 ${s.daily_pnl>=0?'text-accent':'text-danger'}`}>{s.daily_pnl>=0?'+':''}${s.daily_pnl?.toLocaleString()}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Regime</p><p className="text-4xl font-bold mt-2 text-white">{s.regime}</p></Card>
            <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Drawdown</p><p className="text-4xl font-bold mt-2 text-danger">{(s.drawdown*100).toFixed(2)}%</p></Card>
          </div>

          <div className="lg:col-span-8 space-y-8">
            <EquityChart data={d.history || []} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-card p-8">
                <h3 className="text-xs font-bold text-secondary uppercase tracking-widest mb-6 flex items-center gap-2"><Cpu size={14}/> Command Center</h3>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <button onClick={() => start.mutate()} className="btn-elite bg-accent text-black hover:bg-emerald-400">RESUME</button>
                  <button onClick={() => stop.mutate()} className="btn-elite bg-danger text-white hover:bg-red-500">HALT</button>
                </div>
                <div className="flex gap-2 p-2 bg-black/40 rounded-xl border border-white/5">
                  <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="bg-transparent w-16 text-center text-white font-bold outline-none" />
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 text-accent font-bold text-xs hover:text-white">FORCE BUY</button>
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 text-danger font-bold text-xs hover:text-white">FORCE SELL</button>
                </div>
              </div>
              <NewsFeed />
            </div>
          </div>

          <div className="lg:col-span-4 space-y-8">
            <button onClick={() => scan.mutate()} disabled={scan.isPending} className="w-full btn-elite bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center gap-3 py-4 shadow-[0_0_20px_rgba(59,130,246,0.4)]">
              <Radar size={20} className={scan.isPending ? 'animate-spin' : ''} />
              {scan.isPending ? 'ANALYZING...' : 'SCAN MARKET NOW'}
            </button>
            <SignalCard signal={d.signal} />
            <div className="glass-card h-[400px] flex flex-col">
              <div className="p-4 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary tracking-widest">SYSTEM FEED</div>
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
        <div className="glass-card p-10 text-center animate-in fade-in">
          <h2 className="text-2xl font-bold mb-4">Historical Simulation Engine</h2>
          <button onClick={() => backtest.mutate()} className="btn-elite bg-accent text-black">{backtest.isPending ? 'PROCESSING...' : 'RUN 180-DAY BACKTEST'}</button>
          
          {backtest.data && (
            <div className="mt-10 space-y-8">
               <div className="grid grid-cols-4 gap-4">
                 <Card><p className="text-xs text-secondary">Total Return</p><p className="text-2xl font-bold text-accent">{(backtest.data.stats.total_return * 100).toFixed(2)}%</p></Card>
                 <Card><p className="text-xs text-secondary">Sharpe</p><p className="text-2xl font-bold">{backtest.data.stats.sharpe_ratio?.toFixed(2)}</p></Card>
                 <Card><p className="text-xs text-secondary">Drawdown</p><p className="text-2xl font-bold text-danger">{(backtest.data.stats.max_drawdown * 100).toFixed(2)}%</p></Card>
                 <Card><p className="text-xs text-secondary">Trades</p><p className="text-2xl font-bold">{backtest.data.stats.total_trades}</p></Card>
               </div>
               <div className="h-80 w-full bg-black/20 rounded-xl p-4">
                 <ResponsiveContainer><AreaChart data={backtest.data.equity_curve}><YAxis domain={['auto', 'auto']} hide /><Tooltip contentStyle={{background:'#000', border:'1px solid #333'}}/><Area dataKey="value" stroke="#10B981" fill="#10B98122"/></AreaChart></ResponsiveContainer>
               </div>
            </div>
          )}
        </div>
      )}

      {/* LOGS TAB (FIXED SCROLLING) */}
      {tab === 'logs' && (
        <div className="glass-card h-[800px] flex flex-col animate-in fade-in">
           <div className="p-6 border-b border-white/5 font-bold tracking-widest text-secondary">RAW SYSTEM LOGS</div>
           <div className="flex-1 overflow-y-auto p-6 font-mono text-xs space-y-2">
             {logData.map((l:any, i:number) => <div key={i} className="border-b border-white/5 pb-1 opacity-70 hover:opacity-100"><span className={`mr-2 ${l.level==='ERROR'?'text-danger':l.level==='WARNING'?'text-warning':'text-accent'}`}>[{l.level}]</span>{l.message}</div>)}
             <div ref={logEndRef} />
           </div>
        </div>
      )}
    </div>
  );
}
