# 복구 단계 진단

## 관측

- 많은 샘플에서 **`recovery_success_rate=1.0`, `reentry_count=0`**. 현재 스냅샷에서는 복구 루프가 **자주 발동하지 않거나 발동해도 게이트 실패를 역전시키지 못하는** 상태로 보인다.

## 다음 실험

1. 실패 상위 샘플에서 recovery zone 유무·resolved 플래그를 `debug.recovery`에서 추적  
2. 트리거 조건( valid_ratio, None gap, margin ) 과민/둔감 여부  
