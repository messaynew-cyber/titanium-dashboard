import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export function MarketChart({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-500">Waiting for market data...</div>;

  return (
    <div className="card-3d p-6 h-80">
      <h3 className="text-sm font-bold text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
        Live Market Price (GLD)
      </h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="timestamp" hide />
            {/* THIS LINE FIXES THE FLAT GRAPH */}
            <YAxis domain={['dataMin', 'dataMax']} hide />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050505', borderRadius: '8px', border: '1px solid #333' }}
              itemStyle={{ color: '#fff' }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
            />
            <Area 
              type="monotone" 
              dataKey="price" 
              stroke="#3B82F6" 
              strokeWidth={3} 
              fill="url(#colorPrice)" 
              animationDuration={500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
