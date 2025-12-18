import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Globe, Cpu } from 'lucide-react';
import { EquityChart } from './components/dashboard/EquityChart';
import { MarketChart } from './components/dashboard/MarketChart';
import { SignalCard } from './components/dashboard/SignalCard';
import { NewsFeed } from './components/dashboard/NewsFeed';
import { TradeTable } from './components/dashboard/TradeTable';

function useTitanium() {
  const state = useQuery({ queryKey: ['state'], queryFn: () => api.get('/status').then(r => r.data), refetchInterval: 2000 });
  const logs = useQuery({ queryKey: ['logs'], queryFn: () => api.get('/logs?limit=50').then(r => r.data), refetchInterval: 5000 });
  const backtest = useMutation({ mutationFn: () => api.post('/tools/backtest?days=180') });
  const diag = useQuery({ queryKey: ['diag'], queryFn: () => api.get('/tools/diagnostics').then(r => r.data), enabled: false });
  const start = useMutation({ mutationFn: () => api.post('/control/start') });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop') });
  const trade = useMutation({ mutationFn: (p: any) => api.post('/trade/force', p) });
  return { state, logs, backtest, diag, start, stop, trade };
}

export default function App() {
  const { state, logs, backtest, diag, start, stop, trade } = useTitanium();
  const [tab, setTab] = useState('live');
  const [qty, setQty] = useState(10);

  const d = state.data || {};
  const s = d.state || {};
  
  return (
    <div className="min-h-screen p-4 md:p-8 max-w-[1700px] mx-auto">
      {/* HEADER */}
      <header className="flex justify-between items-center mb-10 border-b border-white/5 pb-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3">
            <TrendingUp className="text-accent w-10 h-10" /> TITANIUM <span className="text-accent neon-text">ULTIMATE</span>
          </h1>
          <p className="text-secondary text-xs uppercase tracking-[0.3em] mt-2 opacity-60">Quant-First Asset Management</p>
        </div>
        <div className={`px-8 py-3 rounded-2xl glass-card font-mono text-sm font-bold border ${s.is_active ? 'text-accent border-accent/40 animate-pulse' : 'text-danger border-danger/40'}`}>
          {s.is_active ? '● ENGINE ACTIVE' : '● ENGINE OFFLINE'}
        </div>
      </header>

      {/* TABS NAVIGATION */}
      <nav className="flex gap-4 mb-10">
        {[
          {id: 'live', label: 'LIVE TERMINAL', icon: LayoutDashboard},
          {id: 'sim', label: 'BACKTEST LAB', icon: Play},
          {id: 'health', label: 'DIAGNOSTICS', icon: ShieldCheck},
          {id: 'logs', label: 'CONSOLE', icon: Terminal}
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`btn-elite flex items-center gap-3 text-xs ${tab === t.id ? 'bg-accent text-black shadow-[0_0_30px_rgba(16,185,129,0.4)]' : 'bg-white/5 text-white hover:bg-white/10'}`}>
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </nav>

      {/* CONTENT: LIVE TERMINAL */}
      {tab === 'live' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
          {/* KPI Area */}
          <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="glass-card p-6"><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Equity Value</p><p className="text-3xl font-bold mt-2 text-white neon-text">${s.equity?.toLocaleString()}</p></div>
            <div className="glass-card p-6"><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Target Regime</p><p className="text-3xl font-bold mt-2 text-accent">{s.regime || 'SCANNING'}</p></div>
            <div className="glass-card p-6"><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Drawdown</p><p className="text-3xl font-bold mt-2 text-danger">{(s.drawdown*100).toFixed(2)}%</p></div>
            <div className="glass-card p-6"><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Active Qty</p><p className="text-3xl font-bold mt-2 text-white">{s.position_qty} SHRS</p></div>
          </div>

          <div className="lg:col-span-8 space-y-8">
            <EquityChart data={d.history || []} />
            <MarketChart data={d.history || []} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-card p-8">
                <h3 className="text-xs font-bold text-secondary uppercase tracking-widest mb-6 flex items-center gap-2"><Cpu size={14}/> Command Center</h3>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <button onClick={() => start.mutate()} className="bg-success text-black font-bold py-4 rounded-xl hover:brightness-110 transition active:scale-95">RESUME</button>
                  <button onClick={() => stop.mutate()} className="bg-danger text-white font-bold py-4 rounded-xl hover:brightness-110 transition active:scale-95">HALT</button>
                </div>
                <div className="flex gap-2 p-2 bg-black/40 rounded-xl border border-white/5">
                  <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="bg-transparent w-16 text-center font-bold outline-none" />
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 text-accent font-bold text-xs">BUY</button>
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 text-danger font-bold text-xs">SELL</button>
                </div>
              </div>
              <NewsFeed />
            </div>
          </div>

          <div className="lg:col-span-4 space-y-8">
            <SignalCard signal={d.signal} />
            <div className="glass-card h-[400px] overflow-hidden flex flex-col">
              <div className="p-4 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary tracking-widest">REAL-TIME TELEMETRY</div>
              <div className="flex-1 overflow-y-auto p-5 font-mono text-[10px] space-y-3 opacity-80">
                {logs.data?.slice().reverse().map((l: any, i: number) => (
                  <div key={i} className="flex gap-3"><span className="text-gray-600 shrink-0">{l.timestamp?.split('T')[1]?.split('.')[0]}</span><span className={l.level==='ERROR'?'text-danger':'text-accent'}>[{l.level}]</span><span className="text-gray-300">{l.message}</span></div>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-12">
            <TradeTable trades={d.trades || []} />
          </div>
        </div>
      )}

      {/* OTHER TABS (Simplified for UI consistency) */}
      {tab === 'sim' && (
        <div className="space-y-8 animate-in fade-in duration-500">
          <div className="glass-card p-10 text-center">
            <h2 className="text-2xl font-bold mb-4">Historical Simulation Engine</h2>
            <button onClick={() => backtest.mutate()} className="bg-accent text-black btn-elite">{backtest.isPending ? 'PROCESSING...' : 'RUN 180-DAY BACKTEST'}</button>
            {backtest.data?.data && (
              <div className="mt-10 h-80 w-full">
                <ResponsiveContainer><AreaChart data={backtest.data.data.equity_curve}><YAxis domain={['auto', 'auto']} hide /><Tooltip/><Area dataKey="value" stroke="#10B981" fill="#10B98122"/></AreaChart></ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'health' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in duration-500">
           {diag.data?.map((c: any, i: number) => (
             <div key={i} className="glass-card p-6 flex justify-between items-center"><span className="font-bold text-lg">{c.name}</span><span className={`px-4 py-1 rounded-full text-xs font-bold ${c.status==='PASS'?'bg-success/20 text-success':'bg-danger/20 text-danger'}`}>{c.status}</span></div>
           ))}
           <button onClick={() => diag.refetch()} className="col-span-full btn-elite bg-white/5 text-white">RE-SCAN ALL SYSTEMS</button>
        </div>
      )}

      {tab === 'logs' && (
        <div className="glass-card h-[800px] flex flex-col animate-in fade-in duration-500">
           <div className="p-6 border-b border-white/5 font-bold tracking-widest text-secondary">RAW SYSTEM LOGS</div>
           <div className="flex-1 overflow-y-auto p-6 font-mono text-xs space-y-2">
             {logs.data?.map((l:any, i:number) => <div key={i} className="border-b border-white/5 pb-1 opacity-70 hover:opacity-100"><span className="text-accent mr-2">[{l.level}]</span>{l.message}</div>)}
           </div>
        </div>
      )}
    </div>
  );
}
