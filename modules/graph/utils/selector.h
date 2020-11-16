/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MODULES_GRAPH_UTILS_SELECTOR_H_
#define MODULES_GRAPH_UTILS_SELECTOR_H_

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/foreach.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"

#include "graph/fragment/property_graph_types.h"
#include "graph/utils/error.h"

namespace vineyard {

enum class SelectorType {
  VERTEX_ID,
  VERTEX_DATA,
  EDGE_SRC,
  EDGE_DST,
  EDGE_DATA,
  RESULT
};

class Selector {
 protected:
  explicit Selector(std::string property_name)
      : type_(SelectorType::RESULT), property_name_(std::move(property_name)) {}

  explicit Selector(SelectorType type) : type_(type) {}

 public:
  virtual ~Selector() = default;

  SelectorType type() const { return type_; }

  std::string property_name() const { return property_name_; }

  virtual std::string str() const {
    switch (type_) {
    case SelectorType::VERTEX_ID:
      return "v.id";
    case SelectorType::VERTEX_DATA:
      return "v.data";
    case SelectorType::EDGE_SRC:
      return "e.src";
    case SelectorType::EDGE_DST:
      return "e.dst";
    case SelectorType::EDGE_DATA:
      return "e.data";
    case SelectorType::RESULT: {
      if (property_name_.empty())
        return "r";
      return "r." + property_name_;
    }
    }
  }
  /**
   *
   * @param selector Valid selector pattern:
   * v.id
   * v.data
   * r
   * r.prop_name
   *
   * @return
   */
  static boost::leaf::result<Selector> parse(std::string selector) {
    boost::algorithm::to_lower(selector);
    std::smatch sm;

    std::regex r_vid("v.id");
    std::regex r_vdata("v.data");
    std::regex r_esrc("e.src");
    std::regex r_edst("e.dst");
    std::regex r_edata("e.data");
    std::regex r_result("r");
    std::regex r_result_prop("r.(.*?)");

    if (std::regex_match(selector, sm, r_vid)) {
      return Selector(SelectorType::VERTEX_ID);
    } else if (std::regex_match(selector, sm, r_vdata)) {
      return Selector(SelectorType::VERTEX_DATA);
    } else if (std::regex_match(selector, sm, r_esrc)) {
      return Selector(SelectorType::EDGE_SRC);
    } else if (std::regex_match(selector, sm, r_edst)) {
      return Selector(SelectorType::EDGE_DST);
    } else if (std::regex_match(selector, sm, r_edata)) {
      return Selector(SelectorType::EDGE_DATA);
    } else if (std::regex_match(selector, sm, r_result)) {
      return Selector(SelectorType::RESULT);
    } else if (std::regex_match(selector, sm, r_result_prop)) {
      std::string prop_name = sm[1];
      if (prop_name.empty()) {
        RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                        "Empty property name: " + prop_name);
      }
      return Selector(prop_name);
    }
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "Unmatched selector: " + selector);
  }

  /**
   *
   * @param selectors JSON {"col_name": "selector", ...}
   * @return
   */
  static boost::leaf::result<std::vector<std::pair<std::string, Selector>>>
  parseSelectors(const std::string& s_selectors) {
    std::stringstream ss(s_selectors);
    boost::property_tree::ptree pt;
    std::vector<std::pair<std::string, Selector>> selectors;

    try {
      boost::property_tree::read_json(ss, pt);
      BOOST_FOREACH  // NOLINT(whitespace/parens)
          (boost::property_tree::ptree::value_type & v, pt) {
        CHECK(v.second.empty());
        std::string col_name = v.first;
        std::string s_selector = v.second.data();

        BOOST_LEAF_AUTO(selector, Selector::parse(s_selector));
        selectors.emplace_back(col_name, selector);
      }
    } catch (boost::property_tree::ptree_error& e) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Failed to parse json: " + s_selectors);
    }

    return selectors;
  }

 private:
  SelectorType type_;
  std::string property_name_;
};

class LabeledSelector : public Selector {
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using prop_id_t = property_graph_types::PROP_ID_TYPE;

  LabeledSelector(SelectorType type, label_id_t label_id)
      : Selector(type), label_id_(label_id), property_id_(0) {}

  LabeledSelector(SelectorType type, label_id_t label_id, prop_id_t prop_id)
      : Selector(type), label_id_(label_id), property_id_(prop_id) {}

  LabeledSelector(label_id_t label_id, std::string prop_name)
      : Selector(std::move(prop_name)), label_id_(label_id), property_id_(0) {}

 public:
  label_id_t label_id() const { return label_id_; }

  prop_id_t property_id() const { return property_id_; }

  std::string str() const override {
    switch (type()) {
    case SelectorType::VERTEX_ID:
      return "v.label" + std::to_string(label_id_) + ".id";
    case SelectorType::VERTEX_DATA:
      return "v.label" + std::to_string(label_id_) + ".property" +
             std::to_string(property_id_);
    case SelectorType::EDGE_SRC:
      return "e.label" + std::to_string(label_id_) + ".src";
    case SelectorType::EDGE_DST:
      return "e.label" + std::to_string(label_id_) + ".dst";
    case SelectorType::EDGE_DATA:
      return "e.label" + std::to_string(label_id_) + ".property" +
             std::to_string(property_id_);
    case SelectorType::RESULT: {
      return "r.label" + std::to_string(label_id_) + "." + property_name();
    }
    }
  }

  static boost::leaf::result<LabeledSelector> parse(std::string selector) {
    boost::algorithm::to_lower(selector);
    std::smatch sm;

    /*
     * v.label(0).id
     * v.label(1).property(0)
     * e.label(0).src
     * e.label(1).dst
     * e.label(3).property(2)
     * r.label(4).(prop_name)
     */

    std::regex r_vid("v.label(\\d+).id");
    std::regex r_vdata("v.label(\\d+).property(\\d+)");
    std::regex r_esrc_id("e.label(\\d+).src");
    std::regex r_edst_id("e.label(\\d+).dst");
    std::regex r_edata("e.label(\\d+).property(\\d+)");
    std::regex r_result("r.label(\\d+)");
    std::regex r_result_prop("r.label(\\d+).(.*?)");
    if (std::regex_match(selector, sm, r_vid)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);

      return LabeledSelector(SelectorType::VERTEX_ID, label_id);
    } else if (std::regex_match(selector, sm, r_vdata)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);
      auto prop_id = boost::lexical_cast<prop_id_t>(sm[2]);

      return LabeledSelector(SelectorType::VERTEX_DATA, label_id, prop_id);
    } else if (std::regex_match(selector, sm, r_esrc_id)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);

      return LabeledSelector(SelectorType::EDGE_SRC, label_id);
    } else if (std::regex_match(selector, sm, r_edst_id)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);

      return LabeledSelector(SelectorType::EDGE_DST, label_id);
    } else if (std::regex_match(selector, sm, r_edata)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);
      auto prop_id = boost::lexical_cast<prop_id_t>(sm[2]);

      return LabeledSelector(SelectorType::EDGE_DATA, label_id, prop_id);
    } else if (std::regex_match(selector, sm, r_result)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);
      std::string prop_name = sm[1];

      if (!prop_name.empty()) {
        RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                        "Empty property name: " + prop_name);
      }
      return LabeledSelector(label_id, prop_name);
    } else if (std::regex_match(selector, sm, r_result_prop)) {
      auto label_id = boost::lexical_cast<label_id_t>(sm[1]);
      std::string prop_name = sm[1];

      if (prop_name.empty()) {
        RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                        "Empty property name: " + prop_name);
      }
      return LabeledSelector(label_id, prop_name);
    }
    RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                    "Unmatched selector: " + selector);
  }

  /**
   *
   * @param selectors selectors represented by JSON string e.g. {"col_name":
   * "selector", ...}
   * @return
   */
  static boost::leaf::result<
      std::vector<std::pair<std::string, LabeledSelector>>>
  parseSelectors(const std::string& s_selectors) {
    std::stringstream ss(s_selectors);
    boost::property_tree::ptree pt;
    std::vector<std::pair<std::string, LabeledSelector>> selectors;

    try {
      boost::property_tree::read_json(ss, pt);
      BOOST_FOREACH  // NOLINT(whitespace/parens)
          (boost::property_tree::ptree::value_type & v, pt) {
        CHECK(v.second.empty());
        std::string col_name = v.first;
        std::string s_selector = v.second.data();

        BOOST_LEAF_AUTO(selector, LabeledSelector::parse(s_selector));
        selectors.emplace_back(col_name, selector);
      }
    } catch (boost::property_tree::ptree_error& e) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "Failed to parse json: " + s_selectors);
    }

    return selectors;
  }

 private:
  label_id_t label_id_;
  prop_id_t property_id_;
};

}  // namespace vineyard
#endif  // MODULES_GRAPH_UTILS_SELECTOR_H_
