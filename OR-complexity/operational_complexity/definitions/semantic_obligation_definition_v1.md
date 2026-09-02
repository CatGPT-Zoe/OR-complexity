# Semantic Obligation — Definition v1.0

## 一句话定义

> **Semantic obligation = 一个 family-level predicate，规定哪些 operational decisions 可接受，即可独立检查其是否被正确建模的 feasibility requirement。**

形式化地：

```text
o : ϕ(z_σ(o), Θ) = True
```

其中 `σ(o) ⊆ V_S` 是满足/检查该 obligation 所需读取的 endogenous semantic entity families。

## 三项必须同时满足的条件

一条 rule 成为 semantic obligation，必须同时满足：

### 1. Feasibility-relevant（与可行域相关）

它规定哪些 operational decisions 可接受，即参与定义 intended feasible set。

- ✅ "每个 customer 必须且只能被分配一次。" → 是 obligation
- ❌ "最大化利润。" → objective，不是 feasibility obligation
- ❌ "x ≥ 0"（generic non-negativity）→ domain declaration

### 2. Independently falsifiable（可独立证伪）

可以概念上问："一个 candidate decision 是否违反这一条？" 且可以独立回答。

- ✅ "每个 customer 必须且只能被分配一次。" → 可以独立检查
- ❌ "每个 customer 必须分配一次，并且只能分配给已开启的 facility。" → 应拆成两条，因为可以独立违反

### 3. Family-minimal（按业务 rule template 拆分，不是 scalar index 拆分）

- ✅ `assign_each_customer_once` → 一条 obligation family，不是 `|C|` 条
- ✅ `labor_capacity` 与 `material_capacity` → 即使 support 完全相同（都是 `{production}`），两条不同 rules，**保留两条**

## 明确不是 semantic obligation 的东西

| 类别 | 示例 | 处理方式 |
|---|---|---|
| objective | "maximize profit", "minimize makespan" | 单独保存，不放入 feasibility hypergraph |
| generic variable domain | `x ≥ 0`, `x ∈ {0,1}`, `x integer` | entity metadata，不是 hyperedge |
| solver bookkeeping | Big-M auxiliary, slack variable, solver cut | 不计入 semantic layer |
| pure mathematical helper | epigraph variable, linearization aux | 不计入 semantic layer |
| **但** 实质性业务限值 | "每名工人最多工作 8 小时"、"库存不得超过仓库容量" | 即使 solver 实现为 bound，**仍是 semantic obligation** |

## Split rule（拆分规则）

> 若一条自然语言 rule 包含多个可以独立违反的要求，必须拆成多条 obligations。

**必须拆的典型模式**：

- "X and Y" → 拆成 X 和 Y
- "X only if Y" → 拆成 X 和 obligation(Y) + linking
- "every customer is assigned exactly once and only to an open facility" → 拆成：
  - `assign_each_customer_once`
  - `assignment_requires_open_facility`

## Objective 的处理

Objective 单独保存为：

```json
{
  "sense": "minimize|maximize|satisfy|unknown",
  "description": "...",
  "entity_scope": ["entity_id_1", "entity_id_2"],
  "excluded_from_feasibility_hypergraph": true
}
```

它不进入 coupling metrics，但保留以便未来定义 objective semantic complexity。

## Obligation metadata

每条 obligation 必须保存：

| 字段 | 含义 |
|---|---|
| `support_entity_ids` | 涉及哪些 CSDE families（唯一 primary metric input） |
| `incidence_slots` | 每个 entity 在 obligation 中的角色、index binding、aggregation 信息 |
| `cross_instance` | 是否跨同一 entity 的多个 scalar instance 聚合（如 `sum_j`） |
| `parameter_refs` | 引用了哪些 exogenous parameters |
| `source_spans` | 在原文中的证据位置 |
| `obligation_type` | 预定义类型：capacity, balance, conservation, coverage, assignment, linking, precedence, non_overlap, time_window, logical_implication, service_requirement, resource_limit, policy, other |

## 分类与示例

| 类型 | 示例 | one-entity | multi-entity | cross-instance |
|---|---|---|---|---|
| capacity | `sum_j x_ij ≤ C_i` | 可能 | 通常 | 是 |
| balance | inventory: `I_t = I_{t-1} + P_t - D_t` | 否 | 是 | 是 |
| conservation | flow: `sum_j x_ij = sum_j x_ji` | 可能 | 可能 | 是 |
| coverage | `sum_f X_cf ≥ 1` | 可能 | 可能 | 是 |
| assignment | `sum_f X_cf = 1` | 是 | 否 | 是 |
| linking | `X_cf ≤ Y_f` | 否 | 是 | 否 |
| precedence | `S_j ≥ S_i + p_i` | 否 | 是 | 否 |
| non_overlap | `S_i ≥ S_j + p_j OR S_j ≥ S_i + p_i` | 否 | 是 | 否 |
| time_window | `a_i ≤ S_i ≤ b_i` | 是 | 否 | 否 |

## 与 CSP constraint scope 的关系

Semantic obligation 的 `σ(o)` 对应于 CSP 中的 constraint scope（受该约束限制的变量集合）。区别在于：

- CSP：scope 是 scalar CSP variables
- SOH：scope 是 canonical semantic families

因此 SOH 是 CSP hypergraph 的 semantic abstraction，而不是普通 CSP hypergraph。CSP 理论中的 treewidth / hypergraph width 提供了结构复杂度的数学语言，但不能直接套用 tractability 定理。