import { useState } from 'react';
import { useSystemState, useSystemControl, useLogs } from './hooks/useTitanium';
import { MetricCard } from './components/ui/MetricCard';

function App() {
  const { data: state } = useSystemState();
  const { start, stop, forceTrade } = useSystemControl();
  const { data: logs } = useLogs();
  const [qty, setQty] = useState(1);

  return (
    <div className="min-h-screen bg-background font-sans text-primary p-4 md:p-8">
      {/* HEADER */}
      <header className="mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-primary">TITANIUM DASHBOARD</h1>
          <p className="text-sm text-secondary">Algorithmic Trading System</p>
        </div>
        <div className={`px-4 py-2 rounded-full font-bold text-sm tracking-wide ${state?.is_active ? 'bg-success text-white animate-pulse' : 'bg-danger text-white'}`}>
            {state?.is_active ? '● ONLINE' : '● OFFLINE'}
        </div>
      </header>

      {/* METRICS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Equity" value={`$${state?.equity?.toLocaleString() || 0}`} />
        <MetricCard label="Daily PnL" value={`$${state?.daily_pnl?.toLocaleString() || 0}`} />
        <MetricCard label="Market Regime" value={state?.regime || '-'} />
        <MetricCard label="Drawdown" value={`${((state?.current_drawdown || 0) * 100).toFixed(2)}%`} />
      </div>

      {/* CONTROLS */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="bg-surface border border-border rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-bold mb-4 text-primary">System Power</h3>
          <div className="flex gap-4">
            <button onClick={() => start.mutate()} className="flex-1 bg-success hover:bg-emerald-600 text-white font-bold py-3 rounded transition">START ENGINE</button>
            <button onClick={() => stop.mutate()} className="flex-1 bg-danger hover:bg-red-600 text-white font-bold py-3 rounded transition">STOP ENGINE</button>
          </div>
        </div>

        <div className="bg-surface border border-border rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-bold mb-4 text-primary">Manual Override (GLD)</h3>
          <div className="flex gap-4 items-center">
            <input type="number" value={qty} onChange={(e) => setQty(Number(e.target.value))} className="w-1/3 border border-border rounded p-3 text-center font-bold" />
            <button onClick={() => forceTrade.mutate({ symbol: 'GLD', side: 'buy', qty })} className="flex-1 bg-accent hover:bg-blue-600 text-white font-bold py-3 rounded transition">BUY</button>
            <button onClick={() => forceTrade.mutate({ symbol: 'GLD', side: 'sell', qty })} className="flex-1 bg-secondary hover:bg-slate-600 text-white font-bold py-3 rounded transition">SELL</button>
          </div>
        </div>
      </div>

      {/* LOGS */}
      <div className="bg-surface border border-border rounded-lg shadow-sm h-96 flex flex-col">
        <div className="px-6 py-4 border-b border-border bg-slate-50"><h3 className="text-sm font-bold text-secondary uppercase tracking-wider">System Event Log</h3></div>
        <div className="flex-1 overflow-y-auto p-6 font-mono text-xs space-y-2 bg-white">
          {logs?.slice().reverse().map((l: any, i: number) => (
            <div key={i} className="flex gap-3 border-b border-slate-50 pb-1">
              <span className={l.level === 'ERROR' ? 'text-danger font-bold' : l.level === 'WARNING' ? 'text-warning font-bold' : l.level === 'TRADE' ? 'text-success font-bold' : 'text-accent'}>[{l.level}]</span>
              <span className="text-primary">{l.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
export default App;
