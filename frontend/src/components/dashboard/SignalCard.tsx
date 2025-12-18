import { ArrowUpCircle, ArrowDownCircle, Target, ShieldAlert } from 'lucide-react';

export function SignalCard({ signal }: { signal: any }) {
  if (!signal || !signal.targets) return <div className="animate-pulse card-3d h-64 rounded-lg"></div>;

  const isBullish = signal.sentiment === 'BULLISH';
  const colorClass = isBullish ? 'text-accent' : 'text-danger';
  const bgClass = isBullish ? 'bg-accent/10' : 'bg-danger/10';

  return (
    <div className="card-3d p-6 h-full flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-xs font-bold text-secondary uppercase tracking-wider">AI Signal Generation</h3>
          <span className={`px-3 py-1 rounded text-[10px] font-bold ${bgClass} ${colorClass} border border-${isBullish ? 'accent' : 'danger'}/30`}>
            {signal.sentiment}
          </span>
        </div>
        
        <div className="flex items-center gap-4 mb-6">
          {isBullish ? <ArrowUpCircle size={40} className="text-accent drop-shadow-lg" /> : <ArrowDownCircle size={40} className="text-danger drop-shadow-lg" />}
          <div>
            <p className="text-3xl font-bold text-white tracking-tighter">${signal.targets.entry?.toFixed(2)}</p>
            <p className="text-[10px] text-secondary uppercase">Entry Trigger</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center p-2 bg-black/40 rounded border border-white/5">
            <div className="flex items-center gap-2 text-success">
              <Target size={14} /> <span className="text-[10px] font-bold uppercase">Take Profit</span>
            </div>
            <span className="font-mono font-bold text-success text-sm">${signal.targets.tp?.toFixed(2)}</span>
          </div>
          
          <div className="flex justify-between items-center p-2 bg-black/40 rounded border border-white/5">
            <div className="flex items-center gap-2 text-danger">
              <ShieldAlert size={14} /> <span className="text-[10px] font-bold uppercase">Stop Loss</span>
            </div>
            <span className="font-mono font-bold text-danger text-sm">${signal.targets.sl?.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-white/5">
        <p className="text-[10px] text-secondary mb-1 uppercase font-bold">AI Rationale</p>
        <p className="text-xs text-gray-400 italic">"{signal.reason}"</p>
      </div>
    </div>
  );
}
