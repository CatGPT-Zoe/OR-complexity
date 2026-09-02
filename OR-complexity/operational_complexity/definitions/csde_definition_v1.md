# Canonical Semantic Decision Entity (CSDE) — Definition v1.0

## 一句话定义

> **CSDE = 一个 endogenous（内生）、decision-relevant（决策相关）、具有稳定业务语义与 index signature 的 quantity/relation family。**

给定一个完整 operational instance `I = (T, Θ, R, O)`（`T`=自然语言题面，`Θ`=exogenous sets/parameters/data，`R`=业务/物理/逻辑规则，`O`=objective），CSDE 表示一个 family：

```text
e = (r, I, D, τ, u)
```

其中：

| 字段 | 含义 |
|---|---|
| `r` | semantic role / predicate（语义角色，如 production, shipment, assignment） |
| `I = (I_1, …, I_k)` | canonical index signature（规范索引签名） |
| `D` | semantic value kind（语义取值类型） |
| `τ` | temporal / state semantics（时间/状态语义） |
| `u` | unit / quantity semantics（单位/数量语义） |

**一个 CSDE 是 family，不是 scalar instance。**

例：`production(product, period)` 表示"每产品每期生产量"这一个 endogenous concept，**而不是** `|Product| × |Period|` 个 scalar nodes。

## 判定规则（决定一个 mention 是否成为 CSDE）

一个 mention/概念成为 CSDE，**必须同时满足**：

1. **Endogenous**：问题必须决定或追踪它；它不能是外部给定数据。
2. **Decision-relevant**：它是"要做的决定"或"必须追踪的业务状态"，而非纯派生 helper。
3. **稳定业务语义**：它在业务层面有独立身份，不依赖具体数学编码。
4. **Family-level**：按 index signature 聚合，不按 scalar index 展开。

## 明确**不是** CSDE 的东西

| 类别 | 示例 | 原因 |
|---|---|---|
| exogenous parameters | demand, capacity, cost, distance, profit | 外部给定，非决策 |
| sets / index lists | customer list, periods, facilities | 不属于决策 |
| Big-M helper variable | `u_i` (MTZ subtour) | 仅某一种 formulation 的产物 |
| slack / artificial variable | solver slack | implementation-only |
| epigraph / helper variable | 线性化辅助变量（除非问题本身赋予业务意义） | 数学工具 |
| solver cut / symmetry breaking | 对称性破缺变量 | solver/formulation strengthening |
| aggregation variable | 无独立业务意义的加总变量 | 视语义而定 |

## Endogenous state 是否算 CSDE（必须规范化）

> **只要一个 endogenous state 在业务语义中具有独立身份，并且业务规则或目标直接针对它陈述要求，它就可以成为 CSDE——即使在某种数学 formulation 中它是 derived variable。**

- ✅ `inventory(product, period)`：通常是 CSDE（库存平衡、capacity、safety stock 直接作用于它）。
- ❌ MTZ ordering number：不是 CSDE（"MTZ ordering number"不是业务问题本身要求做出的 operational decision）。
- ⚠️ `load(vehicle, node)`（车载量）：**取决于语义**——
  - 若题目只是"每辆车总配送量不得超过容量"，引入 load 只是 algebraic auxiliary → 不算 CSDE；
  - 若题目描述"车辆在每次 pickup/delivery 后的车载量随路径变化，任一点不得超容" → 这是有独立业务语义的 endogenous state → 算 CSDE。

**判定标准永远是 problem semantics，而不是 solver code 是否声明了变量。**

## Canonicalization（合并规则）

两个文本 mentions `a, b` 合并成一个 CSDE，当且仅当在忽略自然语言别名和 index 符号重命名后，它们表达同一个 endogenous operational relation：

```text
a ∼ b  ⟺  K(a) = K(b),
K(e) = (semantic role, argument roles, time grain, quantity/relation kind)
```

- ✅ `plant production in month t` 与 `monthly amount manufactured at factory` → `production(product, period)`
- ❌ `shipment(origin, destination)` 与 `shipment(origin, destination, period)`：index signature 不同，**不合并**。
- ❌ `facility_open(facility)` 与 `customer_assignment(customer, facility)`：即使某种 MILP encoding 能从一者推导另一者，也是不同 business relations，**不合并**。

## 为什么不能只靠 solver code 恢复 CSDE

Route-based / arc-based / path-based formulations 有时为同一现实 decision 给出完全不同的数学表示。因此 canonicalization 必须**以 business decision ontology / gold semantic annotation 为准**，而不是以 generated variable names 为准。这个 invariance 本身以后应作为实验验证对象（representation stress test）。
