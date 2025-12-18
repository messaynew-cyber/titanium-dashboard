import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { Activity, ShieldCheck, Cpu, Terminal, Play, TrendingUp, LayoutDashboard, Globe } from 'lucide-react';
import { MetricCard } from './components/ui/MetricCard';
import { SignalCard } from './components/dashboard/SignalCard';
import { NewsFeed } from './components/dashboard/NewsFeed';
import { TradeTable } from './components/dashboard/TradeTable';

// HOOKS
function useTitanium() {
  const state = useQuery({ queryKey: ['state'], queryFn: () => api.get('/status').then(r => r.data), refetchInterval: 2000 });
  const logs = useQuery({ queryKey: ['logs'], queryFn: () => api.get('/logs?limit=100').then(r => r.data), refetchInterval: 5000 });
  const start = useMutation({ mutationFn: () => api.post('/control/start') });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop') });
  const trade = useMutation({ mutationFn: (p: any) => api.post('/trade/force', p) });
  const backtest = useMutation({ mutationFn: () => api.post('/tools/backtest?days=180') });
  const diag = useQuery({ queryKey: ['diag'], queryFn: () => api.get('/tools/diagnostics').then(r => r.data) });
  
  return { state, logs, start, stop, trade, backtest, diag };
}

export default function App() {
  const { state, logs, start, stop, trade, backtest, diag } = useTitanium();
  const [tab, setTab] = useState('dash');
  const [qty, setQty] = useState(10);

  const d = state.data || {};
  const s = d.state || {};
  const history = d.history || [];
  const trades = d.trades || [];
  
  // Calculate Log Stats for Visuals
  const logData = logs.data || [];
  const errorCount = logData.filter((l: any) => l.level === 'ERROR').length;
  const warnCount = logData.filter((l: any) => l.level === 'WARNING').length;
  const infoCount = logData.filter((l: any) => l.level === 'INFO').length;
  const logStats = [
    {name: 'INFO', val: infoCount, color: '#10B981'},
    {name: 'WARN', val: warnCount, color: '#F59E0B'},
    {name: 'ERR', val: errorCount, color: '#EF4444'}
  ];

  return (
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-6 max-w-[1600px] mx-auto">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-8 pb-6 border-b border-border gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-3">
            <Activity className="text-accent" /> TITANIUM <span className="text-accent">ULTIMATE</span>
          </h1>
          <p className="text-secondary text-sm mt-1">Institutional Algorithmic Terminal</p>
        </div>
        <div className={`px-4 py-2 rounded font-mono font-bold text-sm tracking-wide border ${s.is_active ? 'bg-accent/10 border-accent text-accent animate-pulse' : 'bg-danger/10 border-danger text-danger'}`}>
            {s.is_active ? '● SYSTEM ONLINE' : '● SYSTEM OFFLINE'}
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <div className="flex gap-4 mb-8 border-b border-border/50 overflow-x-auto">
        {[
          {id: 'dash', icon: LayoutDashboard, label: 'Command Center'},
          {id: 'backtest', icon: Play, label: 'Backtest Lab'},
          {id: 'health', icon: ShieldCheck, label: 'System Diagnostics'},
          {id: 'logs', icon: Terminal, label: 'System Logs'},
        ].map(t => (
          <button 
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`pb-4 px-2 flex items-center gap-2 text-sm font-bold transition-all border-b-2 whitespace-nowrap ${tab === t.id ? 'border-accent text-accent' : 'border-transparent text-secondary hover:text-white'}`}
          >
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      {/* === TAB: COMMAND CENTER === */}
      {tab === 'dash' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard label="Total Equity" value={`$${s.equity?.toLocaleString() || '0.00'}`} />
            <MetricCard label="Daily PnL" value={`$${s.daily_pnl?.toLocaleString() || '0.00'}`} trend={s.daily_pnl >= 0 ? 'up' : 'down'} />
            <MetricCard label="Regime" value={s.regime || 'WAITING'} />
            <MetricCard label="Drawdown" value={`${((s.drawdown || 0) * 100).toFixed(2)}%`} className="border-danger/30" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 space-y-6">
              {/* CHART */}
              <div className="bg-surface border border-border rounded-lg p-6 shadow-sm h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#53db78" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#53db78" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <Tooltip contentStyle={{background: '#051306', border: '1px solid #1f2923', borderRadius: '8px'}} />
                    <Area type="monotone" dataKey="value" stroke="#53db78" fill="url(#g)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 {/* CONTROLS */}
                 <div className="bg-surface border border-border rounded-lg p-6 shadow-sm">
                    <h3 className="font-bold mb-4 flex items-center gap-2 text-sm uppercase text-secondary"><Cpu size={16}/> Controls</h3>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <button onClick={() => start.mutate()} className="bg-success/20 hover:bg-success/30 text-success border border-success/50 py-3 rounded font-bold text-xs">START</button>
                      <button onClick={() => stop.mutate()} className="bg-danger/20 hover:bg-danger/30 text-danger border border-danger/50 py-3 rounded font-bold text-xs">STOP</button>
                    </div>
                    <div className="flex gap-2">
                      <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-20 bg-black/30 border border-border rounded p-2 text-center text-white text-sm" />
                      <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 bg-accent text-black font-bold rounded text-xs hover:brightness-110">BUY</button>
                      <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 bg-white text-black font-bold rounded text-xs hover:bg-gray-200">SELL</button>
                    </div>
                 </div>
                 {/* NEWS */}
                 <NewsFeed />
              </div>
            </div>

            <div className="lg:col-span-4 space-y-6">
              <SignalCard signal={d.signal} />
              <div className="bg-surface border border-border rounded-lg p-4 h-64 overflow-y-auto font-mono text-[10px] space-y-2">
                {logData.slice(0, 20).map((l: any, i: number) => (
                  <div key={i} className="border-b border-border/30 pb-1">
                    <span className={`font-bold mr-2 ${l.level === 'ERROR' ? 'text-danger' : 'text-accent'}`}>[{l.level}]</span>
                    <span className="text-gray-300">{l.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <TradeTable trades={trades} />
        </div>
      )}

      {/* === TAB: BACKTEST LAB === */}
      {tab === 'backtest' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center bg-surface p-6 rounded-lg border border-border">
            <div>
              <h2 className="text-xl font-bold text-white">Strategy Simulation</h2>
              <p className="text-secondary text-sm">Run a 180-day historical simulation with current parameters.</p>
            </div>
            <button 
              onClick={() => backtest.mutate()} 
              disabled={backtest.isPending}
              className="bg-accent text-black px-6 py-3 rounded font-bold hover:brightness-110 disabled:opacity-50"
            >
              {backtest.isPending ? 'RUNNING SIMULATION...' : 'RUN BACKTEST'}
            </button>
          </div>

          {backtest.data?.data && (
            <div className="space-y-6 animate-pulse-slow">
              <div className="grid grid-cols-4 gap-4">
                <MetricCard label="Total Return" value={`${(backtest.data.data.stats.total_return * 100).toFixed(2)}%`} />
                <MetricCard label="Sharpe Ratio" value={backtest.data.data.stats.sharpe_ratio?.toFixed(2)} />
                <MetricCard label="Win Rate" value={`${(backtest.data.data.stats.win_rate * 100).toFixed(1)}%`} />
                <MetricCard label="Max Drawdown" value={`${(backtest.data.data.stats.max_drawdown * 100).toFixed(2)}%`} className="border-danger" />
              </div>
              <div className="bg-surface border border-border rounded-lg p-6 h-[400px]">
                <ResponsiveContainer>
                  <AreaChart data={backtest.data.data.equity_curve}>
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip contentStyle={{background: '#051306', border: '1px solid #1f2923'}} />
                    <Area type="monotone" dataKey="value" stroke="#53db78" fill="#53db78" fillOpacity={0.1} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* === TAB: DIAGNOSTICS === */}
      {tab === 'health' && (
        <div className="space-y-6">
          <div className="flex justify-between">
            <h2 className="text-xl font-bold">System Diagnostics</h2>
            <button onClick={() => diag.refetch()} className="text-accent underline text-sm">Run Full System Scan</button>
          </div>
          <div className="grid grid-cols-1 gap-4">
            {diag.data?.map((c: any, i: number) => (
              <div key={i} className="bg-surface border border-border p-4 rounded-lg flex justify-between items-center">
                <div>
                  <p className="font-bold text-white">{c.name}</p>
                  <p className="text-xs text-secondary">{c.details}</p>
                </div>
                <span className={`px-3 py-1 rounded font-bold text-xs ${c.status === 'PASS' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'}`}>
                  {c.status}
                </span>
              </div>
            ))}
            {!diag.data && <p className="text-secondary text-center py-10">Click scan to check API connections...</p>}
          </div>
        </div>
      )}

      {/* === TAB: LOGS (VISUALIZED) === */}
      {tab === 'logs' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-surface border border-border rounded-lg p-6">
              <h3 className="text-sm font-bold text-secondary uppercase mb-4">Event Distribution</h3>
              <div className="h-40">
                <ResponsiveContainer>
                  <BarChart data={logStats}>
                    <XAxis dataKey="name" stroke="#666" fontSize={10} />
                    <Tooltip cursor={{fill: 'transparent'}} contentStyle={{background: '#000', border: 'none'}}/>
                    <Bar dataKey="val">
                      {logStats.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-surface border border-border rounded-lg p-6 col-span-2 flex flex-col">
              <h3 className="text-sm font-bold text-secondary uppercase mb-4">Full System Log Stream</h3>
              <div className="flex-1 overflow-y-auto font-mono text-xs space-y-2 h-[500px]">
                {logData.map((l: any, i: number) => (
                  <div key={i} className="border-b border-border/30 pb-1">
                    <span className="text-secondary mr-2">{l.timestamp.split('T')[1].split('.')[0]}</span>
                    <span className={`font-bold mr-2 ${l.level === 'ERROR' ? 'text-danger' : l.level === 'WARNING' ? 'text-warning' : 'text-accent'}`}>
                      [{l.level}]
                    </span>
                    <span className="text-white">{l.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
