export function TradeTable({ trades }: { trades: any[] }) {
  return (
    <div className="bg-surface border border-border rounded-lg shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-border bg-slate-50">
        <h3 className="text-sm font-bold text-secondary uppercase tracking-wider">Recent Trades</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-50 text-secondary font-medium">
            <tr>
              <th className="px-6 py-3">Time</th>
              <th className="px-6 py-3">Symbol</th>
              <th className="px-6 py-3">Side</th>
              <th className="px-6 py-3">Qty</th>
              <th className="px-6 py-3">Price</th>
            </tr>
          </thead>
          <tbody>
            {trades && trades.length > 0 ? (
              trades.map((t, i) => (
                <tr key={i} className="border-b border-slate-100 hover:bg-slate-50 transition">
                  <td className="px-6 py-4 font-mono text-xs">{t.time}</td>
                  <td className="px-6 py-4 font-bold">{t.symbol}</td>
                  <td className={`px-6 py-4 font-bold ${t.side === 'BUY' ? 'text-success' : 'text-danger'}`}>{t.side}</td>
                  <td className="px-6 py-4">{t.qty}</td>
                  <td className="px-6 py-4">${t.price.toFixed(2)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-secondary">No trades executed yet today.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
