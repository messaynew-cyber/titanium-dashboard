import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export function EquityChart({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-500">Waiting for data...</div>;

  return (
    <div className="card-3d p-6 h-80">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-bold text-secondary uppercase tracking-wider flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent animate-pulse"></span>
          Live Equity
        </h3>
        <span className="text-xs text-secondary font-mono">USD</span>
      </div>
      
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2923" vertical={false} />
            <XAxis 
              dataKey="timestamp" 
              stroke="#525252" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />
            <YAxis 
              domain={['auto', 'auto']} 
              stroke="#525252" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
              width={40}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050505', borderRadius: '8px', border: '1px solid #333' }}
              itemStyle={{ color: '#fff' }}
              formatter={(value: number) => [`$${value.toLocaleString()}`, 'Equity']}
            />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="#10B981" 
              strokeWidth={2} 
              fill="url(#colorValue)" 
              animationDuration={500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
