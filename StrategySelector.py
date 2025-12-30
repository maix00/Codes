"""
ProductPeriodStrategySelector类用于根据品种（product）和时间段选择特定的复权策略（AdjustmentStrategy）。
支持为每个品种配置多个不同时段的策略，策略以（起始时间，结束时间，策略名称，策略对象）为单位进行管理。
当查询某一品种在特定时间点的策略时，优先返回命中特殊配置的策略，否则返回默认策略（default_strategy）。
主要功能包括：
- add_strategy：为指定品种和时间段添加或更新策略，自动处理时间段重叠和拆分。
- get_strategy：获取指定品种和时间点的策略字典，未命中特殊配置时返回默认策略。
- describe：返回当前所有品种的策略配置描述，便于调试和展示。

"""
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class ProductPeriodStrategySelector:
    """
    根据品种和时间段选择特定的AdjustmentStrategy。
    支持灵活配置每个品种在不同时间段的复权策略，未命中特殊配置时可指定默认策略。
    """

    def __init__(self, default_strategy: Dict[str, Any]):
        # 配置字典: {product: [(start_date, end_date, strategy_dict), ...]}
        self.strategy_map: Dict[str, List[Tuple[datetime, datetime, str, Any]]] = {}
        self.default_strategy = default_strategy

    def add_strategy(
        self,
        product: str,
        start_date: datetime,
        end_date: datetime,
        strategy_name: str,
        strategy: Any,
        interval_unit: str = "seconds"
    ):
        """
        为指定品种和时间段添加特殊策略，包含策略名称
        interval_unit: 时间间隔单位，支持'seconds', 'days'等，默认'seconds'
        """
        if product not in self.strategy_map:
            self.strategy_map[product] = []
        # 新策略时间段
        new_start, new_end = start_date, end_date
        new_name = strategy_name

        # 自适应时间间隔，默认单位为seconds
        delta_kwargs = {interval_unit: 1} if interval_unit else {"seconds": 1}
        try:
            delta_before = timedelta(**delta_kwargs)
            delta_after = timedelta(**delta_kwargs)
        except TypeError:
            # 如果interval_unit不被timedelta支持，回退为seconds
            delta_before = timedelta(seconds=1)
            delta_after = timedelta(seconds=1)

        updated = []
        for entry in self.strategy_map[product]:
            old_start, old_end, old_name, old_strategy = entry
            # 只处理strategy_name相同的条目
            if old_name != new_name:
                updated.append(entry)
                continue

            # 没有重叠
            if old_end < new_start or old_start > new_end:
                updated.append(entry)
                continue

            # 有重叠，拆分或调整
            if old_start < new_start:
                # 保留前段
                updated.append((old_start, new_start - delta_before, old_name, old_strategy))
            if old_end > new_end:
                # 保留后段
                updated.append((new_end + delta_after, old_end, old_name, old_strategy))
            # 被新策略覆盖的部分不保留

        # 添加新策略
        updated.append((new_start, new_end, new_name, strategy))
        # 按起始时间排序
        updated.sort(key=lambda x: x[0])
        self.strategy_map[product] = updated

    def get_strategy(self, product: str, dt: datetime) -> Dict[str, Any]:
        """
        获取指定品种和时间点的策略，优先返回命中特殊配置的策略，否则返回默认策略
        """
        result = {}
        if product in self.strategy_map:
            for start, end, strategy_name, strategy in self.strategy_map[product]:
                if start <= dt <= end:
                    result[strategy_name] = strategy
                if strategy_name not in self.default_strategy:
                    self.default_strategy[strategy_name] = None
        if not result:
            # 如果没有命中，返回默认策略，并补充缺失的strategy_name为None
            return self.default_strategy.copy()
        return result

    def describe(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        返回当前所有配置的描述信息，便于调试和展示
        """
        desc = {}
        for product, lst in self.strategy_map.items():
            desc[product] = []
            for start, end, strategy_name, strategy_obj in lst:
                # 尝试获取 get_name 方法，否则用 strategy_name，否则用类型名
                if hasattr(strategy_obj, 'get_name') and callable(getattr(strategy_obj, 'get_name')):
                    name = strategy_obj.get_name()
                elif strategy_name:
                    name = strategy_name
                else:
                    name = type(strategy_obj).__name__
                desc[product].append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), name))
        return desc
