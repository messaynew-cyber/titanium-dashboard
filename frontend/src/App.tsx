import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from './lib/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { LayoutDashboard, Play, ShieldCheck, Terminal, Zap, TrendingUp, Cpu, Radar, Menu } from 'lucide-react';
import { EquityChart } from './components/dashboard/EquityChart';
import { MarketChart } from './components/dashboard/MarketChart';
import { SignalCard } from './components/dashboard/SignalCard';
import { NewsFeed } from './components/dashboard/NewsFeed';
import { TradeTable } from './components/dashboard/TradeTable';

// HOOKS (Logic remains identical)
function useTitanium() {
  const state = useQuery({ queryKey: ['state'], queryFn: () => api.get('/status').then(r => r.data), refetchInterval: 2000 });
  const logs = useQuery({ queryKey: ['logs'], queryFn: () => api.get('/logs?limit=50').then(r => r.data), refetchInterval: 5000 });
  const backtest = useMutation({ mutationFn: () => api.post('/tools/backtest?days=180') });
  const diag = useQuery({ queryKey: ['diag'], queryFn: () => api.get('/tools/diagnostics').then(r => r.data), enabled: false });
  const start = useMutation({ mutationFn: () => api.post('/control/start') });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop') });
  const trade = useMutation({ mutationFn: (p: any) => api.post('/trade/force', p) });
  const scan = useMutation({ mutationFn: () => api.post('/tools/scan'), onSuccess: () => state.refetch() });
  return { state, logs, backtest, diag, start, stop, trade, scan };
}

// 3D GLASS COMPONENT
const Card = ({ children, className }: any) => (
  <div className={`glass-panel p-6 animate-fade-in-up ${className}`}>
    {children}
  </div>
);

export default function App() {
  const { state, logs, backtest, diag, start, stop, trade, scan } = useTitanium();
  const [tab, setTab] = useState('live');
  const [qty, setQty] = useState(10);

  const d = state.data || {};
  const s = d.state || {};
  
  // Log Stats
  const logData = logs.data || [];
  const logStats = [
    {name: 'INF', val: logData.filter((l:any)=>l.level==='INFO').length, color: '#10B981'},
    {name: 'WRN', val: logData.filter((l:any)=>l.level==='WARNING').length, color: '#F59E0B'},
    {name: 'ERR', val: logData.filter((l:any)=>l.level==='ERROR').length, color: '#EF4444'}
  ];

  return (
    <div className="relative min-h-screen font-sans text-primary selection:bg-accent selection:text-black overflow-hidden">
      
      {/* ANIMATED BACKGROUND BLOBS */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-accent/20 rounded-full blur-[128px] animate-blob"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-blue-600/20 rounded-full blur-[128px] animate-blob animation-delay-2000"></div>
        <div className="absolute top-[40%] left-[40%] w-96 h-96 bg-purple-600/10 rounded-full blur-[128px] animate-blob animation-delay-4000"></div>
      </div>

      <div className="p-4 md:p-8 max-w-[1800px] mx-auto relative z-10">
        
        {/* HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-center mb-8 pb-6 border-b border-white/5 gap-4">
          <div className="text-center md:text-left">
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-white flex items-center justify-center md:justify-start gap-4 drop-shadow-2xl">
              <div className="p-2 bg-accent/10 rounded-xl border border-accent/20">
                <TrendingUp className="text-accent w-8 h-8 md:w-10 md:h-10" />
              </div>
              <span>TITANIUM <span className="text-accent text-glow">X</span></span>
            </h1>
            <p className="text-secondary text-xs font-mono uppercase tracking-[0.4em] mt-3 opacity-70">Autonomous Quant Terminal</p>
          </div>
          
          <div className={`px-6 py-2 rounded-full font-mono text-xs font-bold border backdrop-blur-md flex items-center gap-3 transition-all duration-500 ${s.is_active ? 'bg-accent/10 border-accent text-accent shadow-[0_0_30px_rgba(16,185,129,0.2)]' : 'bg-danger/10 border-danger text-danger'}`}>
            <span className={`w-2 h-2 rounded-full ${s.is_active ? 'bg-accent animate-pulse' : 'bg-danger'}`}></span>
            {s.is_active ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
          </div>
        </header>

        {/* NAVIGATION (SCROLLABLE ON MOBILE) */}
        <nav className="flex gap-2 mb-8 overflow-x-auto pb-2 no-scrollbar">
          {[
            {id: 'live', label: 'COMMAND', icon: LayoutDashboard},
            {id: 'sim', label: 'BACKTEST', icon: Play},
            {id: 'health', label: 'SYSTEM', icon: ShieldCheck},
            {id: 'logs', label: 'LOGS', icon: Terminal}
          ].map(t => (
            <button 
              key={t.id} 
              onClick={() => setTab(t.id)} 
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold text-xs tracking-wider transition-all duration-300 whitespace-nowrap border ${tab === t.id ? 'bg-white/10 border-white/20 text-white shadow-lg' : 'bg-transparent border-transparent text-secondary hover:text-white hover:bg-white/5'}`}
            >
              <t.icon size={16} className={tab === t.id ? 'text-accent' : ''} /> {t.label}
            </button>
          ))}
        </nav>

        {/* === LIVE TERMINAL === */}
        {tab === 'live' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* KPI ROW */}
            <div className="lg:col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Equity</p><p className="text-2xl md:text-4xl font-bold mt-1 text-white text-glow">${s.equity?.toLocaleString() || '0'}</p></Card>
              <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Daily PnL</p><p className={`text-2xl md:text-4xl font-bold mt-1 ${s.daily_pnl>=0?'text-accent text-glow':'text-danger text-glow-red'}`}>{s.daily_pnl>=0?'+':''}${s.daily_pnl?.toLocaleString()}</p></Card>
              <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Regime</p><p className="text-2xl md:text-3xl font-bold mt-1 text-white">{s.regime}</p></Card>
              <Card><p className="text-[10px] text-secondary font-bold tracking-widest uppercase">Drawdown</p><p className="text-2xl md:text-3xl font-bold mt-1 text-danger">{(s.drawdown*100).toFixed(2)}%</p></Card>
            </div>

            {/* MAIN CHART AREA */}
            <div className="lg:col-span-8 space-y-6">
              <EquityChart data={d.history || []} />
              <MarketChart data={d.history || []} />
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <h3 className="text-xs font-bold text-secondary uppercase tracking-widest mb-6 flex items-center gap-2"><Cpu size={14}/> Engine Control</h3>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <button onClick={() => start.mutate()} className="btn-cyber text-accent">RESUME</button>
                    <button onClick={() => stop.mutate()} className="btn-cyber-danger text-danger">HALT</button>
                  </div>
                  <div className="flex gap-2 p-1 bg-black/40 rounded-xl border border-white/5">
                    <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="bg-transparent w-16 text-center text-white font-bold outline-none text-sm" />
                    <button onClick={() => trade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 text-accent font-bold text-[10px] hover:bg-white/5 rounded transition">FORCE BUY</button>
                    <button onClick={() => trade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 text-danger font-bold text-[10px] hover:bg-white/5 rounded transition">FORCE SELL</button>
                  </div>
                </Card>
                <NewsFeed />
              </div>
            </div>

            {/* SIDEBAR */}
            <div className="lg:col-span-4 space-y-6">
              <button 
                onClick={() => scan.mutate()} 
                disabled={scan.isPending}
                className="w-full py-5 rounded-2xl font-bold tracking-widest flex items-center justify-center gap-3 transition-all duration-300 bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_40px_rgba(37,99,235,0.4)] hover:shadow-[0_0_60px_rgba(37,99,235,0.6)] active:scale-95 disabled:opacity-50 disabled:scale-100"
              >
                <Radar size={20} className={scan.isPending ? 'animate-spin' : ''} />
                {scan.isPending ? 'AI ANALYZING...' : 'SCAN MARKET'}
              </button>

              <SignalCard signal={d.signal} />
              
              <div className="glass-panel h-[400px] flex flex-col">
                <div className="p-4 border-b border-white/5 bg-black/30 font-bold text-[10px] text-secondary tracking-widest flex justify-between">
                  <span>LIVE TELEMETRY</span>
                  <span className="flex gap-1"><span className="w-1 h-1 rounded-full bg-red-500"/> <span className="w-1 h-1 rounded-full bg-yellow-500"/> <span className="w-1 h-1 rounded-full bg-green-500"/></span>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-3 opacity-90">
                  {logs.data?.slice().reverse().map((l: any, i: number) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-gray-600 shrink-0">{l.timestamp?.split('T')[1]?.split('.')[0]}</span>
                      <span className={l.level==='ERROR'?'text-danger':'text-accent'}>{l.level === 'ERROR' ? 'ERR' : 'INF'}</span>
                      <span className="text-gray-300 break-words">{l.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="lg:col-span-12">
              <TradeTable trades={d.trades || []} />
            </div>
          </div>
        )}

        {/* === BACKTEST === */}
        {tab === 'sim' && (
          <Card className="flex flex-col items-center justify-center min-h-[500px]">
            <h2 className="text-2xl md:text-3xl font-bold mb-2 text-white">Historical Simulation</h2>
            <p className="text-secondary text-sm mb-8">Run a full 180-day strategy replay using current HMM parameters.</p>
            <button onClick={() => backtest.mutate()} className="btn-cyber text-accent px-10 py-4 text-lg shadow-[0_0_30px_rgba(16,185,129,0.3)]">
              {backtest.isPending ? 'PROCESSING SIMULATION...' : 'INITIATE BACKTEST'}
            </button>
            {backtest.data?.data && (
              <div className="mt-12 h-80 w-full animate-fade-in-up">
                <ResponsiveContainer><AreaChart data={backtest.data.data.equity_curve}><defs><linearGradient id="sim" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/><stop offset="95%" stopColor="#10B981" stopOpacity={0}/></linearGradient></defs><YAxis domain={['auto', 'auto']} hide /><Tooltip contentStyle={{background:'#000', border:'1px solid #333'}}/ ><Area type="monotone" dataKey="value" stroke="#10B981" fill="url(#sim)"/></AreaChart></ResponsiveContainer>
              </div>
            )}
          </Card>
        )}

        {/* === HEALTH === */}
        {tab === 'health' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in-up">
             {diag.data?.map((c: any, i: number) => (
               <div key={i} className="glass-panel p-6 flex justify-between items-center hover:bg-white/5 transition">
                 <span className="font-bold text-lg text-white">{c.name}</span>
                 <span className={`px-3 py-1 rounded text-xs font-bold border ${c.status==='PASS'?'bg-accent/10 border-accent text-accent':'bg-danger/10 border-danger text-danger'}`}>{c.status}</span>
               </div>
             ))}
             <button onClick={() => diag.refetch()} className="col-span-full btn-cyber text-white">RUN FULL SYSTEM DIAGNOSTIC</button>
          </div>
        )}

        {/* === LOGS === */}
        {tab === 'logs' && (
          <Card className="h-[800px] flex flex-col">
             <div className="flex justify-between items-center mb-6">
                <h3 className="font-bold tracking-widest text-secondary">SYSTEM LOGS</h3>
                <div className="flex gap-4 text-xs font-mono">
                  <span className="text-accent">INF: {logStats[0].val}</span>
                  <span className="text-warning">WRN: {logStats[1].val}</span>
                  <span className="text-danger">ERR: {logStats[2].val}</span>
                </div>
             </div>
             <div className="h-32 mb-6">
               <ResponsiveContainer><BarChart data={logStats}><Bar dataKey="val"><Cell fill="#10B981"/><Cell fill="#F59E0B"/><Cell fill="#EF4444"/></Bar></BarChart></ResponsiveContainer>
             </div>
             <div className="flex-1 overflow-y-auto font-mono text-xs space-y-2 p-2 bg-black/20 rounded-xl">
               {logs.data?.map((l:any,i:number) => <div key={i} className="border-b border-white/5 pb-1"><span className={`mr-2 ${l.level==='ERROR'?'text-danger':l.level==='WARNING'?'text-warning':'text-accent'}`}>[{l.level}]</span><span className="text-gray-300">{l.message}</span></div>)}
             </div>
          </Card>
        )}

      </div>
    </div>
  );
}
