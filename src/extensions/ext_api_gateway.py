from typing import Dict, Any

class ext_APIGateway:
    def get_learning_state(self) -> Dict[str, Any]:
        """API patterns and statistics."""
        return {
            'api_statistics': self._api_stats.copy(),
            'endpoint_list': list(self._endpoints.keys())
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """Current request queue and cache."""
        return {
            'request_queue_size': len(self._request_queue),
            'cache_size': len(self._response_cache),
            'rate_limit_status': {
                'current_requests': len(self._rate_limit['request_times']),
                'limit': self._rate_limit['requests_per_minute']
            }
        }
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore API statistics."""
        if 'api_statistics' in state:
            self._api_stats.update(state['api_statistics'])
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Session state restoration not needed for API gateway."""
        pass
