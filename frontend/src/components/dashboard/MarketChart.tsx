import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export function MarketChart({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-500 font-mono text-xs">Waiting for market data...</div>;

  return (
    <div className="card-3d p-6 h-80 relative group">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-bold text-secondary uppercase tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_#3B82F6]"></span>
          Live Market Price (GLD)
        </h3>
        <span className="text-[10px] text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-1 rounded">1M INTERVAL</span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2923" vertical={false} opacity={0.5} />
            <XAxis dataKey="timestamp" hide />
            <YAxis 
              domain={['dataMin', 'dataMax']} 
              stroke="#525252" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
              width={50}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050505', borderRadius: '8px', border: '1px solid #333', color: '#fff' }}
              itemStyle={{ color: '#3B82F6', fontWeight: 'bold' }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
            />
            <Area 
              type="monotone" 
              dataKey="price" 
              stroke="#3B82F6" 
              strokeWidth={2} 
              fill="url(#colorPrice)" 
              animationDuration={0}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
