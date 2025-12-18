import { useState } from 'react';
import { useSystemState, useSystemControl, useLogs } from './hooks/useTitanium';
import { MetricCard } from './components/ui/MetricCard';
import { EquityChart } from './components/dashboard/EquityChart';
import { TradeTable } from './components/dashboard/TradeTable';

function App() {
  const { data: systemData } = useSystemState();
  const { start, stop, forceTrade } = useSystemControl();
  const { data: logs } = useLogs();
  const [qty, setQty] = useState(1);
  const [activeTab, setActiveTab] = useState('dashboard');

  // Safely extract data with defaults
  const state = systemData?.state;
  const history = systemData?.history || [];
  const trades = systemData?.trades || [];

  return (
    <div className="min-h-screen bg-background font-sans text-primary p-4 md:p-8">
      {/* HEADER */}
      <header className="mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary">TITANIUM <span className="text-accent">PRO</span></h1>
          <p className="text-sm text-secondary">Advanced Algorithmic Execution</p>
        </div>
        <div className={`px-4 py-2 rounded-full font-bold text-sm tracking-wide ${state?.is_active ? 'bg-success text-white animate-pulse' : 'bg-danger text-white'}`}>
            {state?.is_active ? '● SYSTEM ONLINE' : '● SYSTEM OFFLINE'}
        </div>
      </header>

      {/* METRICS ROW */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Equity" value={`$${state?.equity?.toLocaleString() || 0}`} />
        <MetricCard label="Daily PnL" value={`$${state?.daily_pnl?.toLocaleString() || 0}`} trend={state?.daily_pnl >= 0 ? 'up' : 'down'} />
        <MetricCard label="Regime" value={state?.regime || '-'} />
        <MetricCard label="Drawdown" value={`${((state?.drawdown || 0) * 100).toFixed(2)}%`} className="border-danger" />
      </div>

      {/* TABS */}
      <div className="flex gap-4 border-b border-border mb-8">
        {['dashboard', 'analytics', 'logs'].map((tab) => (
          <button 
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`pb-2 px-4 font-bold uppercase text-sm tracking-wide transition ${activeTab === tab ? 'text-accent border-b-2 border-accent' : 'text-secondary hover:text-primary'}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* DASHBOARD VIEW */}
      {activeTab === 'dashboard' && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <EquityChart data={history} />
            </div>
            <div className="space-y-4">
              <div className="bg-surface border border-border rounded-lg p-6 shadow-sm">
                 <h3 className="text-sm font-bold text-secondary uppercase mb-4">Manual Override</h3>
                 <div className="flex gap-2 mb-2">
                    <input type="number" value={qty} onChange={(e) => setQty(Number(e.target.value))} className="w-20 border border-border rounded p-2 text-center font-bold" />
                    <button onClick={() => forceTrade.mutate({ symbol: 'GLD', side: 'buy', qty })} className="flex-1 bg-success hover:bg-emerald-600 text-white font-bold rounded">BUY</button>
                    <button onClick={() => forceTrade.mutate({ symbol: 'GLD', side: 'sell', qty })} className="flex-1 bg-danger hover:bg-red-600 text-white font-bold rounded">SELL</button>
                 </div>
                 <div className="flex gap-2 mt-4 pt-4 border-t border-border">
                    <button onClick={() => start.mutate()} className="flex-1 bg-primary text-white py-2 rounded font-bold text-xs">START SYSTEM</button>
                    <button onClick={() => stop.mutate()} className="flex-1 border border-danger text-danger py-2 rounded font-bold text-xs hover:bg-rose-50">STOP</button>
                 </div>
              </div>
            </div>
          </div>
          <TradeTable trades={trades} />
        </div>
      )}

      {/* LOGS VIEW */}
      {activeTab === 'logs' && (
        <div className="bg-surface border border-border rounded-lg shadow-sm h-[600px] flex flex-col">
          <div className="px-6 py-4 border-b border-border bg-slate-50"><h3 className="text-sm font-bold text-secondary uppercase">System Internals</h3></div>
          <div className="flex-1 overflow-y-auto p-6 font-mono text-xs space-y-2 bg-white">
            {logs?.slice().reverse().map((l: any, i: number) => (
              <div key={i} className="flex gap-3 border-b border-slate-50 pb-1">
                <span className={l.level === 'ERROR' ? 'text-danger font-bold' : l.level === 'WARNING' ? 'text-warning font-bold' : l.level === 'TRADE' ? 'text-success font-bold' : 'text-accent'}>[{l.level}]</span>
                <span className="text-primary">{l.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* ANALYTICS VIEW */}
      {activeTab === 'analytics' && (
        <div className="text-center py-20 text-secondary bg-surface border border-border rounded-lg">
          <p className="font-bold">Advanced Analytics Module</p>
          <p className="text-sm mt-2">Data is accumulating. Alpha/Beta metrics require 7 days of trading history.</p>
        </div>
      )}
    </div>
  );
}
export default App;
