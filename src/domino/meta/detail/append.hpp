#ifndef DOMINO_META_DETAIL_APPEND_HPP_
#define DOMINO_META_DETAIL_APPEND_HPP_

#include <domino/meta/detail/list.hpp>
#include <domino/meta/detail/utility.hpp>

namespace domino {

namespace meta {

namespace detail {

template <class... L>
struct meta_append_impl;

template <
    class L1 = meta_list<>, class L2 = meta_list<>, class L3 = meta_list<>,
    class L4 = meta_list<>, class L5 = meta_list<>, class L6 = meta_list<>,
    class L7 = meta_list<>, class L8 = meta_list<>, class L9 = meta_list<>,
    class L10 = meta_list<>, class L11 = meta_list<>>
struct append_11_impl {};

template <
    template <class...> class L1, class... T1, template <class...> class L2,
    class... T2, template <class...> class L3, class... T3,
    template <class...> class L4, class... T4, template <class...> class L5,
    class... T5, template <class...> class L6, class... T6,
    template <class...> class L7, class... T7, template <class...> class L8,
    class... T8, template <class...> class L9, class... T9,
    template <class...> class L10, class... T10, template <class...> class L11,
    class... T11>

struct append_11_impl<L1<T1...>, L2<T2...>, L3<T3...>, L4<T4...>, L5<T5...>,
                      L6<T6...>, L7<T7...>, L8<T8...>, L9<T9...>, L10<T10...>,
                      L11<T11...>> {
  using type = L1<T1..., T2..., T3..., T4..., T5..., T6..., T7..., T8..., T9...,
                  T10..., T11...>;
};

template <

    class L00 = meta_list<>, class L01 = meta_list<>, class L02 = meta_list<>,
    class L03 = meta_list<>, class L04 = meta_list<>, class L05 = meta_list<>,
    class L06 = meta_list<>, class L07 = meta_list<>, class L08 = meta_list<>,
    class L09 = meta_list<>, class L0A = meta_list<>, class L10 = meta_list<>,
    class L11 = meta_list<>, class L12 = meta_list<>, class L13 = meta_list<>,
    class L14 = meta_list<>, class L15 = meta_list<>, class L16 = meta_list<>,
    class L17 = meta_list<>, class L18 = meta_list<>, class L19 = meta_list<>,
    class L20 = meta_list<>, class L21 = meta_list<>, class L22 = meta_list<>,
    class L23 = meta_list<>, class L24 = meta_list<>, class L25 = meta_list<>,
    class L26 = meta_list<>, class L27 = meta_list<>, class L28 = meta_list<>,
    class L29 = meta_list<>, class L30 = meta_list<>, class L31 = meta_list<>,
    class L32 = meta_list<>, class L33 = meta_list<>, class L34 = meta_list<>,
    class L35 = meta_list<>, class L36 = meta_list<>, class L37 = meta_list<>,
    class L38 = meta_list<>, class L39 = meta_list<>, class L40 = meta_list<>,
    class L41 = meta_list<>, class L42 = meta_list<>, class L43 = meta_list<>,
    class L44 = meta_list<>, class L45 = meta_list<>, class L46 = meta_list<>,
    class L47 = meta_list<>, class L48 = meta_list<>, class L49 = meta_list<>,
    class L50 = meta_list<>, class L51 = meta_list<>, class L52 = meta_list<>,
    class L53 = meta_list<>, class L54 = meta_list<>, class L55 = meta_list<>,
    class L56 = meta_list<>, class L57 = meta_list<>, class L58 = meta_list<>,
    class L59 = meta_list<>, class L60 = meta_list<>, class L61 = meta_list<>,
    class L62 = meta_list<>, class L63 = meta_list<>, class L64 = meta_list<>,
    class L65 = meta_list<>, class L66 = meta_list<>, class L67 = meta_list<>,
    class L68 = meta_list<>, class L69 = meta_list<>, class L70 = meta_list<>,
    class L71 = meta_list<>, class L72 = meta_list<>, class L73 = meta_list<>,
    class L74 = meta_list<>, class L75 = meta_list<>, class L76 = meta_list<>,
    class L77 = meta_list<>, class L78 = meta_list<>, class L79 = meta_list<>,
    class L80 = meta_list<>, class L81 = meta_list<>, class L82 = meta_list<>,
    class L83 = meta_list<>, class L84 = meta_list<>, class L85 = meta_list<>,
    class L86 = meta_list<>, class L87 = meta_list<>, class L88 = meta_list<>,
    class L89 = meta_list<>, class L90 = meta_list<>, class L91 = meta_list<>,
    class L92 = meta_list<>, class L93 = meta_list<>, class L94 = meta_list<>,
    class L95 = meta_list<>, class L96 = meta_list<>, class L97 = meta_list<>,
    class L98 = meta_list<>, class L99 = meta_list<>, class LA0 = meta_list<>,
    class LA1 = meta_list<>, class LA2 = meta_list<>, class LA3 = meta_list<>,
    class LA4 = meta_list<>, class LA5 = meta_list<>, class LA6 = meta_list<>,
    class LA7 = meta_list<>, class LA8 = meta_list<>, class LA9 = meta_list<>

    >
struct append_111_impl {
  using type = typename append_11_impl<

      typename append_11_impl<L00, L01, L02, L03, L04, L05, L06, L07, L08, L09,
                              L0A>::type,
      typename append_11_impl<meta_list<>, L10, L11, L12, L13, L14, L15, L16,
                              L17, L18, L19>::type,
      typename append_11_impl<meta_list<>, L20, L21, L22, L23, L24, L25, L26,
                              L27, L28, L29>::type,
      typename append_11_impl<meta_list<>, L30, L31, L32, L33, L34, L35, L36,
                              L37, L38, L39>::type,
      typename append_11_impl<meta_list<>, L40, L41, L42, L43, L44, L45, L46,
                              L47, L48, L49>::type,
      typename append_11_impl<meta_list<>, L50, L51, L52, L53, L54, L55, L56,
                              L57, L58, L59>::type,
      typename append_11_impl<meta_list<>, L60, L61, L62, L63, L64, L65, L66,
                              L67, L68, L69>::type,
      typename append_11_impl<meta_list<>, L70, L71, L72, L73, L74, L75, L76,
                              L77, L78, L79>::type,
      typename append_11_impl<meta_list<>, L80, L81, L82, L83, L84, L85, L86,
                              L87, L88, L89>::type,
      typename append_11_impl<meta_list<>, L90, L91, L92, L93, L94, L95, L96,
                              L97, L98, L99>::type,
      typename append_11_impl<meta_list<>, LA0, LA1, LA2, LA3, LA4, LA5, LA6,
                              LA7, LA8, LA9>::type

      >::type;
};

template <

    class L00, class L01, class L02, class L03, class L04, class L05, class L06,
    class L07, class L08, class L09, class L0A, class L10, class L11, class L12,
    class L13, class L14, class L15, class L16, class L17, class L18, class L19,
    class L20, class L21, class L22, class L23, class L24, class L25, class L26,
    class L27, class L28, class L29, class L30, class L31, class L32, class L33,
    class L34, class L35, class L36, class L37, class L38, class L39, class L40,
    class L41, class L42, class L43, class L44, class L45, class L46, class L47,
    class L48, class L49, class L50, class L51, class L52, class L53, class L54,
    class L55, class L56, class L57, class L58, class L59, class L60, class L61,
    class L62, class L63, class L64, class L65, class L66, class L67, class L68,
    class L69, class L70, class L71, class L72, class L73, class L74, class L75,
    class L76, class L77, class L78, class L79, class L80, class L81, class L82,
    class L83, class L84, class L85, class L86, class L87, class L88, class L89,
    class L90, class L91, class L92, class L93, class L94, class L95, class L96,
    class L97, class L98, class L99, class LA0, class LA1, class LA2, class LA3,
    class LA4, class LA5, class LA6, class LA7, class LA8, class LA9,
    class... Lr

    >
struct append_inf_impl {
  using prefix = typename append_111_impl<

      L00, L01, L02, L03, L04, L05, L06, L07, L08, L09, L0A, L10, L11, L12, L13,
      L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24, L25, L26, L27, L28,
      L29, L30, L31, L32, L33, L34, L35, L36, L37, L38, L39, L40, L41, L42, L43,
      L44, L45, L46, L47, L48, L49, L50, L51, L52, L53, L54, L55, L56, L57, L58,
      L59, L60, L61, L62, L63, L64, L65, L66, L67, L68, L69, L70, L71, L72, L73,
      L74, L75, L76, L77, L78, L79, L80, L81, L82, L83, L84, L85, L86, L87, L88,
      L89, L90, L91, L92, L93, L94, L95, L96, L97, L98, L99, LA0, LA1, LA2, LA3,
      LA4, LA5, LA6, LA7, LA8, LA9

      >::type;

  using type = typename meta_append_impl<prefix, Lr...>::type;
};

template <class... L>
struct meta_append_impl
    : meta_if_c<(sizeof...(L) > 111), meta_quote<append_inf_impl>,
                meta_if_c<(sizeof...(L) > 11), meta_quote<append_111_impl>,
                          meta_quote<append_11_impl>>>::template fn<L...> {};

}  // namespace detail

template <class... L>
using meta_append = typename detail::meta_append_impl<L...>::type;

}  // namespace meta

}  // namespace domino

#endif  // DOMINO_META_DETAIL_APPEND_HPP_
