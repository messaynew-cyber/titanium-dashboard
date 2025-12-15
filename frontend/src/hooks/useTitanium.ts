import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';

export function useSystemState() {
  return useQuery({
    queryKey: ['systemState'],
    queryFn: async () => { const { data } = await api.get('/status'); return data; },
    refetchInterval: 2000,
  });
}
export function useLogs() {
  return useQuery({
    queryKey: ['logs'],
    queryFn: async () => { const { data } = await api.get('/logs?limit=50'); return data; },
    refetchInterval: 5000,
  });
}
export function useSystemControl() {
  const queryClient = useQueryClient();
  const start = useMutation({ mutationFn: () => api.post('/control/start'), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemState'] }) });
  const stop = useMutation({ mutationFn: () => api.post('/control/stop'), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemState'] }) });
  const forceTrade = useMutation({ mutationFn: (p: any) => api.post('/trade/force', p), onSuccess: () => alert("Trade Executed!") });
  return { start, stop, forceTrade };
}
