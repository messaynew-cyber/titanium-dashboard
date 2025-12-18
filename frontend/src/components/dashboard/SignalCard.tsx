import { ArrowUpCircle, ArrowDownCircle, Target, ShieldAlert, TrendingUp } from 'lucide-react';

export function SignalCard({ signal }: { signal: any }) {
  if (!signal || !signal.targets) return <div className="animate-pulse bg-surface h-64 rounded-lg"></div>;

  const isBullish = signal.sentiment === 'BULLISH';
  const colorClass = isBullish ? 'text-accent' : 'text-danger';
  const bgClass = isBullish ? 'bg-accent/10' : 'bg-danger/10';

  return (
    <div className="bg-surface border border-border rounded-lg p-6 shadow-sm h-full flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-sm font-bold text-secondary uppercase tracking-wider">AI Signal Generation</h3>
          <span className={`px-3 py-1 rounded text-xs font-bold ${bgClass} ${colorClass} border border-${isBullish ? 'accent' : 'danger'}/30`}>
            {signal.sentiment}
          </span>
        </div>
        
        <div className="flex items-center gap-4 mb-6">
          {isBullish ? <ArrowUpCircle size={48} className="text-accent" /> : <ArrowDownCircle size={48} className="text-danger" />}
          <div>
            <p className="text-3xl font-bold text-white">${signal.targets.entry?.toFixed(2)}</p>
            <p className="text-xs text-secondary">Current Reference Price</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center p-3 bg-black/20 rounded border border-border/50">
            <div className="flex items-center gap-2 text-success">
              <Target size={16} /> <span className="text-xs font-bold uppercase">Take Profit</span>
            </div>
            <span className="font-mono font-bold text-success">${signal.targets.tp?.toFixed(2)}</span>
          </div>
          
          <div className="flex justify-between items-center p-3 bg-black/20 rounded border border-border/50">
            <div className="flex items-center gap-2 text-danger">
              <ShieldAlert size={16} /> <span className="text-xs font-bold uppercase">Stop Loss</span>
            </div>
            <span className="font-mono font-bold text-danger">${signal.targets.sl?.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-border">
        <p className="text-xs text-secondary mb-1 uppercase font-bold">AI Rationale</p>
        <p className="text-sm text-gray-300 italic">"{signal.reason}"</p>
      </div>
    </div>
  );
}
