import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export function EquityChart({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-500 font-mono text-xs">INITIALIZING DATA STREAM...</div>;

  return (
    <div className="card-3d p-6 h-80 relative group">
      {/* HEADER */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-bold text-secondary uppercase tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_8px_#10B981]"></span>
          Live Equity Performance
        </h3>
        <div className="flex items-center gap-2">
           <span className="text-[10px] text-accent bg-accent/10 border border-accent/20 px-2 py-1 rounded">REAL-TIME</span>
        </div>
      </div>
      
      {/* CHART AREA */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2923" vertical={false} opacity={0.5} />
            <XAxis 
              dataKey="timestamp" 
              stroke="#525252" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />
            <YAxis 
              domain={['dataMin', 'dataMax']} 
              stroke="#525252" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
              width={40}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050505', borderRadius: '8px', border: '1px solid #1f2923', color: '#fff' }}
              itemStyle={{ color: '#10B981', fontWeight: 'bold' }}
              labelStyle={{ color: '#666', fontSize: '10px', marginBottom: '5px' }}
              formatter={(value: number) => [`$${value.toLocaleString(undefined, {minimumFractionDigits: 2})}`, 'Equity']}
            />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="#10B981" 
              strokeWidth={2} 
              fill="url(#colorEquity)" 
              animationDuration={500}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
