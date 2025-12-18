import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, ShieldCheck, Cpu, Terminal, Play, Square, TrendingUp, AlertTriangle } from 'lucide-react';

// --- DATA HOOKS ---
function useTitanium() {
  const state = useQuery({ queryKey: ['state'], queryFn: () => api.get('/status').then(r => r.data), refetchInterval: 2000 });
  const logs = useQuery({ queryKey: ['logs'], queryFn: () => api.get('/logs').then(r => r.data), refetchInterval: 5000 });
  const start = useMutation({ mutationFn: () => api.post('/control/start') });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop') });
  const trade = useMutation({ mutationFn: (p: any) => api.post('/trade/force', p) });
  const backtest = useMutation({ mutationFn: () => api.post('/tools/backtest?days=180') });
  const diag = useQuery({ queryKey: ['diag'], queryFn: () => api.get('/tools/diagnostics').then(r => r.data), enabled: false });
  
  return { state, logs, start, stop, trade, backtest, diag };
}

// --- COMPONENTS ---
const Card = ({ children, className }: any) => <div className={`bg-surface border border-border rounded-lg p-5 shadow-sm ${className}`}>{children}</div>;
const Badge = ({ txt, color }: any) => <span className={`px-2 py-1 rounded text-xs font-bold bg-${color}-500/20 text-${color}-400 border border-${color}-500/30`}>{txt}</span>;

export default function App() {
  const { state, logs, start, stop, trade, backtest, diag } = useTitanium();
  const [tab, setTab] = useState('dash');
  const [qty, setQty] = useState(10);

  const d = state.data || {};
  const s = d.state || {};
  
  return (
    <div className="min-h-screen bg-background text-primary font-sans p-6">
      {/* HEADER */}
      <header className="flex justify-between items-center mb-8 pb-6 border-b border-border">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-3">
            <Activity className="text-accent" /> TITANIUM <span className="text-accent">PRO</span>
          </h1>
          <p className="text-secondary text-sm mt-1">Autonomous Hedge Fund Terminal</p>
        </div>
        <div className="flex gap-4">
           <div className="text-right">
             <p className="text-xs text-secondary">SYSTEM STATUS</p>
             <p className={`font-mono font-bold ${s.is_active ? 'text-accent' : 'text-danger'}`}>
               {s.is_active ? '● OPERATIONAL' : '● OFFLINE'}
             </p>
           </div>
        </div>
      </header>

      {/* TABS */}
      <div className="flex gap-6 mb-8 border-b border-border/50">
        {[
          {id: 'dash', icon: TrendingUp, label: 'Live Desk'},
          {id: 'backtest', icon: Play, label: 'Backtest Lab'},
          {id: 'health', icon: ShieldCheck, label: 'System Health'},
          {id: 'logs', icon: Terminal, label: 'Raw Logs'}
        ].map(t => (
          <button 
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`pb-4 flex items-center gap-2 text-sm font-bold transition-all border-b-2 ${tab === t.id ? 'border-accent text-accent' : 'border-transparent text-secondary hover:text-white'}`}
          >
            <t.icon size={16} /> {t.label}
          </button>
        ))}
      </div>

      {/* DASHBOARD TAB */}
      {tab === 'dash' && (
        <div className="space-y-6">
          {/* KPI ROW */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <p className="text-xs text-secondary uppercase">Total Equity</p>
              <p className="text-3xl font-bold text-white mt-1">${s.equity?.toLocaleString() || '0.00'}</p>
            </Card>
            <Card>
              <p className="text-xs text-secondary uppercase">Daily PnL</p>
              <p className={`text-3xl font-bold mt-1 ${s.daily_pnl >= 0 ? 'text-accent' : 'text-danger'}`}>
                {s.daily_pnl >= 0 ? '+' : ''}${s.daily_pnl?.toLocaleString() || '0.00'}
              </p>
            </Card>
            <Card>
              <p className="text-xs text-secondary uppercase">Active Regime</p>
              <p className="text-3xl font-bold text-white mt-1">{s.regime || 'WAITING'}</p>
            </Card>
            <Card>
              <p className="text-xs text-secondary uppercase">Cash Reserve</p>
              <p className="text-3xl font-bold text-white mt-1">${s.cash?.toLocaleString() || '0.00'}</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* CHART */}
            <Card className="lg:col-span-2 h-[400px]">
              <div className="h-full w-full">
                <ResponsiveContainer>
                  <AreaChart data={d.history || []}>
                    <defs>
                      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#53db78" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#53db78" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="timestamp" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip contentStyle={{background: '#051306', border: '1px solid #1f2923', borderRadius: '8px'}} />
                    <Area type="monotone" dataKey="value" stroke="#53db78" fill="url(#g)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* CONTROLS */}
            <div className="space-y-4">
              <Card>
                <h3 className="font-bold mb-4 flex items-center gap-2"><Cpu size={16}/> Engine Control</h3>
                <div className="grid grid-cols-2 gap-2">
                  <button onClick={() => start.mutate()} className="bg-accent/20 hover:bg-accent/30 text-accent border border-accent/50 py-3 rounded font-bold transition">ENGAGE</button>
                  <button onClick={() => stop.mutate()} className="bg-danger/20 hover:bg-danger/30 text-danger border border-danger/50 py-3 rounded font-bold transition">KILL SWITCH</button>
                </div>
              </Card>

              <Card>
                <h3 className="font-bold mb-4 flex items-center gap-2"><TrendingUp size={16}/> Force Execution</h3>
                <div className="flex gap-2">
                  <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-20 bg-black/30 border border-border rounded p-2 text-center text-white" />
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 bg-accent text-black font-bold rounded hover:brightness-110">BUY</button>
                  <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 bg-white text-black font-bold rounded hover:bg-gray-200">SELL</button>
                </div>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* BACKTEST TAB */}
      {tab === 'backtest' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold">Strategy Simulation (180 Days)</h2>
            <button 
              onClick={() => backtest.mutate()} 
              disabled={backtest.isPending}
              className="bg-accent text-black px-6 py-2 rounded font-bold hover:brightness-110 disabled:opacity-50"
            >
              {backtest.isPending ? 'RUNNING SIMULATION...' : 'RUN BACKTEST'}
            </button>
          </div>

          {backtest.data?.data && (
            <div className="space-y-6 animate-pulse-slow">
              <div className="grid grid-cols-4 gap-4">
                <Card><p className="text-secondary text-xs">Total Return</p><p className="text-2xl font-bold text-accent">{(backtest.data.data.stats.total_return * 100).toFixed(2)}%</p></Card>
                <Card><p className="text-secondary text-xs">Sharpe Ratio</p><p className="text-2xl font-bold text-white">{backtest.data.data.stats.sharpe_ratio.toFixed(2)}</p></Card>
                <Card><p className="text-secondary text-xs">Win Rate</p><p className="text-2xl font-bold text-white">{(backtest.data.data.stats.win_rate * 100).toFixed(1)}%</p></Card>
                <Card><p className="text-secondary text-xs">Max Drawdown</p><p className="text-2xl font-bold text-danger">{(backtest.data.data.stats.max_drawdown * 100).toFixed(2)}%</p></Card>
              </div>
              <Card className="h-[300px]">
                <ResponsiveContainer>
                  <AreaChart data={backtest.data.data.equity_curve}>
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip contentStyle={{background: '#051306', border: '1px solid #1f2923'}} />
                    <Area type="monotone" dataKey="value" stroke="#53db78" fill="#53db78" fillOpacity={0.1} />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </div>
          )}
        </div>
      )}

      {/* HEALTH TAB */}
      {tab === 'health' && (
        <div className="space-y-6">
          <div className="flex justify-between">
            <h2 className="text-xl font-bold">System Diagnostics</h2>
            <button onClick={() => diag.refetch()} className="text-accent underline text-sm">Refresh Diagnostics</button>
          </div>
          <div className="grid grid-cols-1 gap-4">
            {diag.data?.map((c: any, i: number) => (
              <Card key={i} className="flex justify-between items-center">
                <div>
                  <p className="font-bold text-white">{c.name}</p>
                  <p className="text-xs text-secondary">{c.details}</p>
                </div>
                <Badge txt={c.status} color={c.status === 'PASS' ? 'green' : 'red'} />
              </Card>
            ))}
            {!diag.data && <p className="text-secondary">Click refresh to run diagnostics...</p>}
          </div>
        </div>
      )}

      {/* LOGS TAB */}
      {tab === 'logs' && (
        <Card className="h-[600px] overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto font-mono text-xs space-y-2">
            {logs.data?.slice().reverse().map((l: any, i: number) => (
              <div key={i} className="border-b border-border/30 pb-1">
                <span className="text-secondary mr-2">{l.timestamp.split('T')[1].split('.')[0]}</span>
                <span className={`font-bold mr-2 ${l.level === 'ERROR' ? 'text-danger' : l.level === 'WARNING' ? 'text-warning' : 'text-accent'}`}>
                  [{l.level}]
                </span>
                <span className="text-white">{l.message}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
