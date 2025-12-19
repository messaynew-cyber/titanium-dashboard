import { useEffect, useState } from 'react';
import { ExternalLink, Newspaper } from 'lucide-react';

export function NewsFeed() {
  const [news, setNews] = useState<any[]>([]);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        // Using a public RSS to JSON bridge
        const res = await fetch('https://api.rss2json.com/v1/api.json?rss_url=https://finance.yahoo.com/news/rssindex');
        const data = await res.json();
        setNews(data.items.slice(0, 5));
      } catch (e) { console.error(e); }
    };
    fetchNews();
  }, []);

  return (
    <div className="card-3d h-full flex flex-col">
      <div className="px-6 py-4 border-b border-white/5 bg-black/20">
        <h3 className="text-xs font-bold text-secondary uppercase tracking-wider flex items-center gap-2">
          <Newspaper size={14} /> Market Intelligence
        </h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {news.map((item, i) => (
          <a key={i} href={item.link} target="_blank" rel="noreferrer" className="block group">
            <div className="flex gap-4 items-start">
              <div className="flex-1">
                <h4 className="text-xs font-bold text-white group-hover:text-accent transition leading-snug mb-1">
                  {item.title}
                </h4>
                <div className="flex items-center gap-1 mt-1 text-[10px] text-gray-500">
                  <span>{new Date(item.pubDate).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                  <span className="text-gray-700">•</span>
                  <span className="flex items-center gap-1 group-hover:text-accent transition">Read <ExternalLink size={8}/></span>
                </div>
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}
