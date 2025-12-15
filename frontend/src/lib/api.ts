import axios from 'axios';
const API_URL = "/api/v1";
export const api = axios.create({ baseURL: API_URL, headers: { "Content-Type": "application/json" } });
