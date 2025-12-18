import { useState } from 'react';
import { useSystemState, useSystemControl, useLogs } from './hooks/useTitanium';
import { MetricCard } from './components/ui/MetricCard';
import { EquityChart } from './components/dashboard/EquityChart';
import { TradeTable } from './components/dashboard/TradeTable';
import { SignalCard } from './components/dashboard/SignalCard'; // NEW
import { NewsFeed } from './components/dashboard/NewsFeed';     // NEW
import { Activity, ShieldCheck, Cpu, Terminal, Play, TrendingUp } from 'lucide-react';

// --- HOOKS ---
// (Same hooks as before, just ensuring we export what we use)
function useTitanium() {
  const { data: systemData, refetch } = useSystemState();
  const logs = useLogs();
  const control = useSystemControl();
  return { state: systemData, logs, control, refresh: refetch };
}

export default function App() {
  const { state: systemData, logs, control } = useTitanium();
  const [qty, setQty] = useState(10);
  const [tab, setTab] = useState('dash');

  const s = systemData?.state || {};
  const history = systemData?.history || [];
  const trades = systemData?.trades || [];
  const signal = systemData?.signal || {};

  return (
    <div className="min-h-screen bg-background text-primary font-sans p-4 md:p-6 max-w-[1600px] mx-auto">
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-8 pb-6 border-b border-border gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-3">
            <Activity className="text-accent" /> TITANIUM <span className="text-accent">ELITE</span>
          </h1>
          <p className="text-secondary text-sm mt-1">Institutional Grade Trading Terminal</p>
        </div>
        <div className={`px-4 py-2 rounded font-mono font-bold text-sm tracking-wide border ${s.is_active ? 'bg-accent/10 border-accent text-accent animate-pulse' : 'bg-danger/10 border-danger text-danger'}`}>
            {s.is_active ? '● SYSTEM ONLINE' : '● SYSTEM OFFLINE'}
        </div>
      </header>

      {/* METRICS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Equity" value={`$${s.equity?.toLocaleString() || '0.00'}`} />
        <MetricCard label="Daily PnL" value={`$${s.daily_pnl?.toLocaleString() || '0.00'}`} trend={s.daily_pnl >= 0 ? 'up' : 'down'} />
        <MetricCard label="Active Regime" value={s.regime || 'WAITING'} />
        <MetricCard label="Drawdown" value={`${((s.drawdown || 0) * 100).toFixed(2)}%`} className="border-danger/30" />
      </div>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
        
        {/* LEFT COL: CHART (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          <EquityChart data={history} />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
             {/* MANUAL CONTROLS */}
             <div className="bg-surface border border-border rounded-lg p-6 shadow-sm">
                <h3 className="font-bold mb-4 flex items-center gap-2 text-sm uppercase text-secondary"><Cpu size={16}/> Command Center</h3>
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <button onClick={() => control.start.mutate()} className="bg-success/20 hover:bg-success/30 text-success border border-success/50 py-3 rounded font-bold text-xs transition">START ENGINE</button>
                  <button onClick={() => control.stop.mutate()} className="bg-danger/20 hover:bg-danger/30 text-danger border border-danger/50 py-3 rounded font-bold text-xs transition">KILL SWITCH</button>
                </div>
                <div className="flex gap-2">
                  <input type="number" value={qty} onChange={e => setQty(Number(e.target.value))} className="w-20 bg-black/30 border border-border rounded p-2 text-center text-white text-sm" />
                  <button onClick={() => control.forceTrade.mutate({symbol: 'GLD', side: 'buy', qty})} className="flex-1 bg-accent text-black font-bold rounded text-xs hover:brightness-110">FORCE BUY</button>
                  <button onClick={() => control.forceTrade.mutate({symbol: 'GLD', side: 'sell', qty})} className="flex-1 bg-white text-black font-bold rounded text-xs hover:bg-gray-200">FORCE SELL</button>
                </div>
             </div>

             {/* NEWS FEED */}
             <NewsFeed />
          </div>
        </div>

        {/* RIGHT COL: SIGNAL CARD & LOGS (4 cols) */}
        <div className="lg:col-span-4 space-y-6 flex flex-col">
          <SignalCard signal={signal} />
          
          <div className="bg-surface border border-border rounded-lg shadow-sm flex-1 flex flex-col min-h-[300px]">
            <div className="px-6 py-4 border-b border-border bg-black/20 flex justify-between items-center">
              <h3 className="text-sm font-bold text-secondary uppercase flex items-center gap-2">
                <Terminal size={16} /> Live Logs
              </h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-2 bg-black/10">
              {logs.data?.slice().reverse().map((l: any, i: number) => (
                <div key={i} className="border-b border-border/30 pb-1">
                  <span className="text-gray-500 mr-2">{l.timestamp.split('T')[1].split('.')[0]}</span>
                  <span className={`font-bold mr-2 ${l.level === 'ERROR' ? 'text-danger' : l.level === 'WARNING' ? 'text-warning' : 'text-accent'}`}>
                    [{l.level}]
                  </span>
                  <span className="text-gray-300">{l.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>

      {/* FOOTER: TRADES */}
      <TradeTable trades={trades} />
    </div>
  );
}
