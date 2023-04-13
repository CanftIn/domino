#ifndef DOMINO_LIST_H_
#define DOMINO_LIST_H_

struct list_head {
  struct list_head *next, *prev;
};

#define LIST_HEAD_INIT(name) { &(name), &(name) }

#define LIST_HEAD(name) \
  struct list_head name = LIST_HEAD_INIT(name)

static inline void INIT_LIST_HEAD(struct list_head *list) {
  list->next = list;
  list->prev = list;
}

static inline void __list_add(struct list_head *node,
                              struct list_head *prev,
                              struct list_head *next) {
  next->prev = node;
  node->next = next;
  node->prev = prev;
  prev->next = node;
}

static inline void list_add(struct list_head *node,
                            struct list_head *head) {
  __list_add(node, head, head->next);
}

static inline void list_add_tail(struct list_head *node,
                                 struct list_head *head) {
  __list_add(node, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next) {
  next->prev = prev;
  prev->next = next;
}

static inline void list_del(struct list_head *entry) {
  __list_del(entry->prev, entry->next);
}

static inline void list_move(struct list_head *list, struct list_head *head) {
  __list_del(list->prev, list->next);
  list_add(list, head);
}

static inline void list_move_tail(struct list_head *list,
                                  struct list_head *head) {
  __list_del(list->prev, list->next);
  list_add_tail(list, head);
}

static inline int list_empty(const struct list_head *head) {
  return head->next == head;
}

static inline void __list_splice(struct list_head *list,
                                 struct list_head *head) {
  struct list_head *first = list->next;
  struct list_head *last = list->prev;
  struct list_head *at = head->next;

  first->prev = head;
  head->next = first;

  last->next = at;
  at->prev = last;
}

static inline void list_splice(struct list_head *list, struct list_head *head) {
  if (!list_empty(list))
    __list_splice(list, head);
}

static inline void list_splice_init(struct list_head *list,
                                    struct list_head *head) {
  if (!list_empty(list)) {
    __list_splice(list, head);
    INIT_LIST_HEAD(list);
  }
}

#define list_entry(ptr, type, member) \
  ((type*)((char *)(ptr) - (unsigned long)(&((type *)0)->member)))

#define list_for_each(pos, head) \
  for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_prev(pos, head) \
  for (pos = (head)->prev; pos != (head); pos = pos->prev)

#define list_for_each_safe(pos, n, head) \
  for (pos = (head)->next, n = pos->next; pos != head; \
    pos = n, n = pos->next)

#define list_for_each_entry(pos, head, member) \
  for (pos = list_entry((head)->next, typeof(*pos), member); \
    &pos->member != (head); \
    pos = list_entry(pos->member.next, typeof(*pos), member))

struct slist_node {
  struct slist_node *next;
};



#endif  // DOMINO_LIST_H_