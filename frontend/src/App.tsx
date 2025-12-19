
import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { Activity, ShieldCheck, Cpu, Terminal, Play, TrendingUp, LayoutDashboard, Zap } from 'lucide-react';
import { SignalCard } from './components/dashboard/SignalCard';
import { NewsFeed } from './components/dashboard/NewsFeed';
import { TradeTable } from './components/dashboard/TradeTable';
import { MarketChart } from './components/dashboard/MarketChart'; // NEW

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

// 3D CARD COMPONENT
const Card = ({ children, className }: any) => <div className={`card-3d p-6 ${className}`}>{children}</div>;

export default function App() {
  const { state, logs, start, stop, trade, backtest, diag } = useTitanium();
  const [tab, setTab] = useState('dash');
  const [qty, setQty] = useState(10);

  const d = state.data || {};
  const s = d.state || {};
  const history = d.history || [];
  
  // Prepare Log Stats
  const logData = logs.data || [];
  const logStats = [
    {name: 'INFO', val: logData.filter((l:any)=>l.level==='INFO').length, color: '#10B981'},
    {name: 'WARN', val: logData.filter((l:any)=>l.level==='WARNING').length, color: '#F59E0B'},
    {name: 'ERR', val: logData.filter((l:any)=>l.level==='ERROR').length, color: '#EF4444'}
  ];

  return (
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-6 max-w-[1800px] mx-auto selection:bg-accent selection:text-black">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-10 pb-6 border-b border-white/5 gap-4">
        <div>
          <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3 drop-shadow-lg">
            <Activity className="text-accent w-8 h-8" /> TITANIUM <span className="text-accent text-glow">X</span>
          </h1>
          <p className="text-secondary text-sm mt-1 tracking-widest uppercase">Next-Gen Quantitative Terminal</p>
        </div>
        <div className={`px-6 py-2 rounded-full font-mono font-bold text-sm tracking-wide border btn-3d ${s.is_active ? 'bg-accent/10 border-accent text-accent animate-pulse shadow-[0_0_20px_rgba(16,185,129,0.3)]' : 'bg-danger/10 border-danger text-danger'}`}>
            {s.is_active ? '● SYSTEM ONLINE' : '● SYSTEM OFFLINE'}
        </div>
      </header>

      {/* NAV */}
      <div className="flex gap-4 mb-10 border-b border-white/5 overflow-x-auto pb-1">
        {[
          {id: 'dash', icon: LayoutDashboard, label: 'COMMAND'},
          {id: 'backtest', icon: Play, label: 'SIMULATION'},
          {id: 'health', icon: ShieldCheck, label: 'DIAGNOSTICS'},
          {id: 'logs', icon: Terminal, label: 'CONSOLE'},
        ].map(t => (
          <button 
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`pb-4 px-4 flex items-center gap-2 text-xs font-bold tracking-widest transition-all border-b-2 whitespace-nowrap hover:text-white ${tab === t.id ? 'border-accent text-accent text-glow' : 'border-transparent text-secondary'}`}
          >
            <t.icon size={14} /> {t.label}
          </button>
        ))}
      </div>

      {/* === DASHBOARD === */}
      {tab === 'dash' && (
        <div className="space-y-8 animate-in fade-in duration-500">
          
          {/* KPI METRICS (3D CARDS) */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <p className="text-[10px] text-secondary uppercase tracking-widest font-bold">Total Equity</p>
              <p className="text-4xl font-bold text-white mt-2 text-glow">${s.equity?.toLocaleString() || '0.00'}</p>
            </Card>
            <Card>
              <p className="text-[10px] text-secondary uppercase tracking-widest font-bold">Daily PnL</p>
              <p className={`text-4xl font-bold mt-2 ${s.daily_pnl >= 0 ? 'text-accent text-glow' : 'text-danger text-glow-red'}`}>
                {s.daily_pnl >= 0 ? '+' : ''}${s.daily_pnl?.toLocaleString() || '0.00'}
              </p>
            </Card>
            <Card>
              <p className="text-[10px] text-secondary uppercase tracking-widest font-bold">Regime</p>
              <p className="text-4xl font-bold text-white mt-2">{s.regime || 'SCANNING'}</p>
            </Card>
            <Card>
              <p className="text-[10px] text-secondary uppercase tracking-widest font-bold">Drawdown</p>
              <p className="text-4xl font-bold text-danger mt-2 text-glow-red">{(s.drawdown * 100).toFixed(2)}%</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 space-y-6">
              {/* MAIN CHART (Equity) */}
              <div className="card-3d p-6 h-80">
                <h3 className="text-sm font-bold text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-accent animate-pulse"></span> Live Equity Performance
                </h3>
                <div className="h-64 w-full">
                  <ResponsiveContainer>
                    <AreaChart data={history}>
                      <defs>
                        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.4}/>
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <Tooltip contentStyle={{background: '#0a0a0a', border: '1px solid #333', borderRadius: '8px'}} />
                      <Area type="monotone" dataKey="value" stroke="#10B981" strokeWidth={3} fill="url(#g)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* NEW MARKET CHART */}
              <MarketChart data={history} />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 {/* CONTROLS */}
                 <Card>
                    <h3 className="font-bold mb-6 flex items-center gap-2 text-sm uppercase text-secondary"><Cpu size={16}/> Engine Control</h3>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <button onClick={() => start.mutate()} className="btn-3d bg-accent hover:bg-emerald-400 text-black py-4 rounded-lg font-bold text-sm flex items-center justify-center gap-2"><Zap size={16}/> START</button>
                      <button onClick={() => stop.mutate()} className="btn-3d bg-danger hover:bg-red-500 text-white py-4 rounded-lg font-bold text-sm">STOP</button>
                    </div>
                    <div className="flex gap-2 p-2 bg-black/30 rounded-lg border border-white/5">
                      <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-16 bg-transparent text-center text-white font-bold outline-none" />
                      <div className="w-[1px] bg-white/10 mx-2"></div>
                      <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 text-accent font-bold text-xs hover:text-white transition">FORCE BUY</button>
                      <div className="w-[1px] bg-white/10 mx-2"></div>
                      <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 text-danger font-bold text-xs hover:text-white transition">FORCE SELL</button>
                    </div>
                 </Card>
                 <NewsFeed />
              </div>
            </div>

            {/* SIDEBAR */}
            <div className="lg:col-span-4 space-y-6">
              <SignalCard signal={d.signal} />
              
              <div className="card-3d flex flex-col h-[400px]">
                <div className="p-4 border-b border-white/5 bg-black/20 flex justify-between items-center">
                  <h3 className="text-xs font-bold text-secondary uppercase flex items-center gap-2">
                    <Terminal size={14} /> System Feed
                  </h3>
                  <div className="flex gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500"></div>
                    <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  </div>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-3">
                  {logData.slice(0, 20).map((l: any, i: number) => (
                    <div key={i} className="flex gap-2 opacity-80 hover:opacity-100 transition">
                      <span className="text-gray-600">{l.timestamp.split('T')[1].split('.')[0]}</span>
                      <span className={`${l.level === 'ERROR' ? 'text-danger' : 'text-accent'}`}>
                        {l.level === 'ERROR' ? 'ERR' : 'INF'}
                      </span>
                      <span className="text-gray-300">{l.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <TradeTable trades={d.trades} />
        </div>
      )}

      {/* OTHER TABS (Keeping functional but simple) */}
      {tab === 'backtest' && (
        <Card className="h-[600px] flex flex-col items-center justify-center">
           <button onClick={() => backtest.mutate()} className="btn-3d bg-accent text-black px-8 py-4 rounded-xl font-bold text-lg hover:scale-105 transition">
             {backtest.isPending ? 'SIMULATING...' : 'INITIATE BACKTEST SEQUENCE'}
           </button>
           {backtest.data && <div className="mt-8 text-accent font-mono">Sim Complete. Data Loaded.</div>}
        </Card>
      )}
      
      {tab === 'health' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
           {diag.data?.map((c: any, i: number) => (
             <Card key={i} className="flex justify-between items-center">
               <span className="text-white font-bold">{c.name}</span>
               <span className={`text-xs px-2 py-1 rounded ${c.status==='PASS'?'bg-accent/20 text-accent':'bg-danger/20 text-danger'}`}>{c.status}</span>
             </Card>
           ))}
           <button onClick={() => diag.refetch()} className="col-span-2 btn-3d bg-white/10 py-3 rounded text-white font-bold">RUN DIAGNOSTICS</button>
        </div>
      )}

      {tab === 'logs' && (
        <Card className="h-[800px]">
           <ResponsiveContainer height={200}>
             <BarChart data={logStats}><Bar dataKey="val"><Cell fill="#10B981"/><Cell fill="#F59E0B"/><Cell fill="#EF4444"/></Bar></BarChart>
           </ResponsiveContainer>
           <div className="h-[500px] overflow-y-auto font-mono text-xs mt-4 space-y-2">
             {logData.map((l:any, i:number) => <div key={i} className="border-b border-white/5 pb-1"><span className="text-accent mr-2">[{l.level}]</span>{l.message}</div>)}
           </div>
        </Card>
      )}

    </div>
  );
}
