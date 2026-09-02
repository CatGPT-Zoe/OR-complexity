# Semantic Layer vs. Algebraic Layer Boundary — v1.0

## 双层结构

```text
Operational specification → H_S (semantic obligation hypergraph) → OC_HG(S)
chosen mathematical formulation → H_A (algebraic hypergraph) → OC_HG(A)
```

- **Semantic layer** `H_S = (V_S, E_S, σ, ρ)`：只依赖自然语言、数据语义和 business rules，**不依赖**具体 MILP/CP/solver encoding。这是 problem-intrinsic 的。
- **Algebraic layer** `H_A = (V_A, C_A, σ_A)`：与具体 chosen formulation 相关，用于测量 representation inflation、auxiliary-variable coupling、linearization 等，**不等于** intrinsic operational complexity。

## 元素计入规则（一张表定死）

| 元素 | Semantic node | Semantic obligation | Algebraic layer | 规则 |
|---|---|---|---|---|
| 实际生产量、shipment、assignment、route choice | 是 | — | 是 | endogenous operational concept |
| inventory / completion 等真实 endogenous state | 通常是 | — | 是 | 前提：业务语义直接使用 |
| demand、capacity、cost、distance | 否 | 否 | — | parameter，exogenous |
| index set / customer list / periods | 否 | 否 | — | set，不属于 decision |
| objective | 否 | 核心图中否 | — | 单独保存，objective 排序 feasible decisions |
| capacity rule | — | 是 | 是 | substantive feasibility condition |
| time window | — | 是 | — | substantive business rule |
| "最多工作 8 小时" | — | 是 | — | 即使实现为 bound 也算；语义决定，不按代码语法 |
| generic nonnegativity `x ≥ 0` | entity attribute | 否 domain/bound | — | representation/domain declaration |
| binary/integer declaration | entity attribute | 否 domain | — | domain 不作为 coupling edge |
| Big-M auxiliary | 否 | 否 | 是 | implementation-only |
| MTZ order variable | 通常否 | 否 | 是 | routing formulation artifact |
| slack variable | 否 | 否 | 是 | implementation-only |
| epigraph/helper variable | 通常否 | 否 | 是 | 除非问题本身赋予业务意义 |
| solver cut / symmetry breaking | 否 | 否 | 是 | solver/formulation strengthening |
| aggregation variable | 视语义 | 视语义 | 是 | 有独立业务意义才进入 semantic |
| 同一 business rule 的不同 scalar rows | — | 一个 family edge | 多 scalar rows | family-level collapse |
| 两个不同 rules、相同 support | — | 两个 edges | 两个 families | multi-hypergraph 保留 |

## Mapping

```text
π : V_A → 2^{V_S} ∪ {⊥}
π : C_A → 2^{O_S} ∪ {⊥}
```

`⊥` 表示"这个 algebraic element 没有独立 semantic counterpart"，例如：

```text
u_i (MTZ) ↦ ⊥
```

这正是 representation stress-test 所需的结构：MIPLIB-NL 中 `aggregation definitions`、`logical implication`、`domain/integrality` 等结构出现于 solver model，但不等同于问题的 semantic obligations。

## 关键推论

- **C_S 不随 formulation 改变；C_F 可以改变。**
  - 例：一个 operational rule 原本只有三个 semantic entities，但 MILP linearization 引入五个 binary auxiliary families。`C_S` 不应因此变化，`C_F` 应该变化。
  - 这正是双层 complexity 的体现，也是 representation stress-test（F1/F2/F3）的理论基础。
- Objective 虽不进入 feasibility-coupling core，但不能从 annotation schema 中删除；保存 `O_obj = (sense, semantic objective, V)` 供未来单独定义 objective semantic complexity。