/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/torch.h>
#include <optional>
#include <string>

namespace torchrec::url_parser {

struct Authority {
  std::string username;
  std::string password;
};

struct Url {
  std::optional<Authority> authority;
  std::string host;
  std::optional<uint16_t> port;
  std::optional<std::string> param;
};

inline Authority parse_authority(std::string_view authority_str) {
  Authority authority;

  auto colon_pos = authority_str.find(':');
  if (colon_pos != std::string_view::npos) {
    authority.username = authority_str.substr(0, colon_pos);
    authority.password = authority_str.substr(colon_pos + 1);
  } else {
    // only username
    authority.username = authority_str;
  }
  return authority;
}

inline Url parse_url(std::string_view url_str) {
  Url url;
  // (username (":" password)? "@")? host ":" port ("/" | "/?" param)?
  // Assume there will only be one '@'
  auto at_pos = url_str.find('@');
  if (at_pos != std::string_view::npos) {
    Authority authority = parse_authority(url_str.substr(0, at_pos));
    url.authority = authority;
    url_str = url_str.substr(at_pos + 1);
  }
  // There should be no '/' in host:port.
  auto slash_pos = url_str.find('/');
  std::string_view host_port_str;
  if (slash_pos != std::string_view::npos) {
    host_port_str = url_str.substr(0, slash_pos);
    url_str = url_str.substr(slash_pos + 1);
  } else {
    host_port_str = url_str;
    url_str = "";
  }

  auto colon_pos = host_port_str.find(':');
  if (colon_pos != std::string_view::npos) {
    url.host = host_port_str.substr(0, colon_pos);
    auto port_str = host_port_str.substr(colon_pos + 1);
    url.port = std::stoi(std::string(port_str));
  } else {
    url.host = host_port_str;
  }

  if (!url_str.empty()) {
    if (url_str[0] != '?') {
      throw std::invalid_argument("invalid parameter: " + std::string(url_str));
    } else {
      url.param = url_str.substr(1);
    }
  }

  return url;
}

} // namespace torchrec::url_parser
