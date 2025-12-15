import { useSystemState, useSystemControl, useLogs } from './hooks/useTitanium';
import { MetricCard } from './components/ui/MetricCard';

function App() {
  const { data: state } = useSystemState();
  const { start, stop } = useSystemControl();
  const { data: logs } = useLogs();

  return (
    <div className="min-h-screen bg-background font-sans text-primary p-8">
      <header className="mb-8 flex justify-between">
        <h1 className="text-xl font-bold">TITANIUM DASHBOARD</h1>
        <div className={`px-3 py-1 rounded ${state?.is_active ? 'bg-success' : 'bg-danger'} text-white`}>
            {state?.is_active ? 'ONLINE' : 'OFFLINE'}
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Equity" value={`$${state?.equity || 0}`} />
        <MetricCard label="Daily PnL" value={`$${state?.daily_pnl || 0}`} />
        <MetricCard label="Regime" value={state?.regime || '-'} />
        <MetricCard label="Drawdown" value={`${((state?.current_drawdown || 0) * 100).toFixed(2)}%`} />
      </div>

      <div className="flex gap-4 mb-8">
        <button onClick={() => start.mutate()} className="bg-success text-white px-4 py-2 rounded">START</button>
        <button onClick={() => stop.mutate()} className="bg-danger text-white px-4 py-2 rounded">STOP</button>
      </div>

      <div className="bg-surface border border-border rounded p-4 h-64 overflow-y-auto font-mono text-xs">
        {logs?.map((l: any, i: number) => <div key={i}>[{l.level}] {l.message}</div>)}
      </div>
    </div>
  );
}
export default App;
