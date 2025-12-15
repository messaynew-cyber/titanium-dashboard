import { clsx } from 'clsx';
export function MetricCard({ label, value, subValue, trend, className }: any) {
  const trendColor = trend === 'up' ? 'text-success' : trend === 'down' ? 'text-danger' : 'text-secondary';
  return (
    <div className={clsx("bg-surface border border-border rounded-lg p-5 shadow-sm", className)}>
      <p className="text-sm font-medium text-secondary uppercase tracking-wider">{label}</p>
      <div className="mt-2 flex items-baseline">
        <span className={clsx("text-2xl font-bold tracking-tight text-primary", trendColor)}>{value}</span>
      </div>
    </div>
  );
}