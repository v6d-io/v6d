/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#include "server/util/kubectl.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/bind.hpp"
#include "boost/filesystem.hpp"
#include "boost/process.hpp"

#include "common/util/asio.h"

namespace vineyard {

Kubectl::Kubectl(asio::io_context& context) : proc_(new Process(context)) {
  proc_->Start("kubectl", {"apply", "-f", "-"},
               [](Status const&, const std::string&) { return Status::OK(); });
}

Kubectl::~Kubectl() { proc_->Terminate(); }

void Kubectl::Apply(const std::string& content, callback_t<> callback) {
  proc_->AsyncWrite(content, [this, callback](Status const& status) {
    // TODO: improve the error diagnostic
    if (!status.ok()) {
      for (auto const& line : Diagnostic()) {
        VLOG(10) << "kubectl: " << line;
      }
    }
    return callback(status);
  });
}

static std::string generate_local_object(
    std::map<InstanceID, std::string> const& instances, const json& object) {
  std::string object_id = object["id"].get_ref<std::string const&>();
  std::string signature =
      SignatureToString(object["signature"].get<Signature>());
  std::string type_name = object["typename"].get_ref<std::string const&>();

  std::string job_name = "none";
  if (object.contains("JOB_NAME")) {
    job_name = object["JOB_NAME"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable JOB_NAME not set" << std::endl;
  }
  std::string client_pod_name = "none-set";
  if (object.contains("POD_NAME")) {
    client_pod_name = object["POD_NAME"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable POD_NAME not set" << std::endl;
  }
  std::string client_pod_namespace = "default";
  if (object.contains("POD_NAMESPACE")) {
    client_pod_namespace =
        object["POD_NAMESPACE"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable POD_NAMESPACE not set" << std::endl;
  }

  InstanceID instance_id = object["instance_id"].get<InstanceID>();
  std::string vineyardd_name = getenv("VINEYARDD_NAME");
  std::string namespace_ = getenv("VINEYARDD_NAMESPACE");
  std::string uid = getenv("VINEYARDD_UID");

  /* clang-format off */
  std::string crd = "\n"
                    "\n---"
                    "\n"
                    "\napiVersion: k8s.v6d.io/v1alpha1"
                    "\nkind: LocalObject"
                    "\nmetadata:"
                    "\n  name: " + object_id +
                    "\n  namespace: " + namespace_ +
                    "\n  labels:"
                    "\n    k8s.v6d.io/signature: " + signature +
                    "\n    job: " + job_name +
                    "\n    created-by-podname: " + client_pod_name +
                    "\n    created-by-podnamespace: " + client_pod_namespace +
                    "\n  ownerReferences:"
                    "\n    - apiVersion: k8s.v6d.io/v1alpha1"
                    "\n      kind: Vineyardd"
                    "\n      name: " + vineyardd_name +
                    "\n      uid: " + uid +
                    "\nspec:"
                    "\n  id: " + object_id +
                    "\n  signature: " + signature +
                    "\n  typename: " + type_name +
                    "\n  instance_id: " + std::to_string(instance_id) +
                    "\n  hostname: " + instances.at(instance_id) +
                    "\n  metadata: " + type_name +
                    "\n\n";
  /* clang-format on */
  return crd;
}

static std::string generate_global_object(
    std::map<InstanceID, std::string> const& instances, const json& object) {
  std::string object_id = object["id"].get_ref<std::string const&>();
  std::string signature =
      SignatureToString(object["signature"].get<Signature>());
  std::string type_name = object["typename"].get_ref<std::string const&>();

  std::string job_name = "none";
  if (object.contains("JOB_NAME")) {
    job_name = object["JOB_NAME"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable JOB_NAME not set" << std::endl;
  }
  std::string client_pod_name = "none-set";
  if (object.contains("POD_NAME")) {
    client_pod_name = object["POD_NAME"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable POD_NAME not set" << std::endl;
  }
  std::string client_pod_namespace = "default";
  if (object.contains("POD_NAMESPACE")) {
    client_pod_namespace =
        object["POD_NAMESPACE"].get_ref<std::string const&>();
  } else {
    std::cout << "Environment variable POD_NAMESPACE not set" << std::endl;
  }

  std::vector<std::string> members;

  std::string vineyardd_name = getenv("VINEYARDD_NAME");
  std::string namespace_ = getenv("VINEYARDD_NAMESPACE");
  std::string uid = getenv("VINEYARDD_UID");

  std::string crds;
  for (auto const& kv : object.items()) {
    if (kv.value().is_object() && !kv.value().empty()) {
      crds += generate_local_object(instances, kv.value());
      members.emplace_back(
          SignatureToString(kv.value()["signature"].get<Signature>()));
    }
  }

  /* clang-format off */
  std::string crd = "\n"
                    "\n---"
                    "\n"
                    "apiVersion: k8s.v6d.io/v1alpha1"
                    "\nkind: GlobalObject"
                    "\nmetadata:"
                    "\n  name: " + object_id +
                    "\n  namespace: " + namespace_ +
                    "\n  labels:"
                    "\n    job: " + job_name +
                    "\n    created-by-podname: " + client_pod_name +
                    "\n    created-by-podnamespace: " + client_pod_namespace +
                    "\n  ownerReferences:"
                    "\n    - apiVersion: k8s.v6d.io/v1alpha1"
                    "\n      kind: Vineyardd"
                    "\n      name: " + vineyardd_name +
                    "\n      uid: " + uid +
                    "\nspec:"
                    "\n  id: " + object_id +
                    "\n  signature: " + signature +
                    "\n  typename: " + type_name +
                    "\n  metadata: " + type_name +
                    "\n  members:";
  /* clang-format on */

  for (auto const& sig : members) {
    crd += "\n  - " + sig;
  }
  crd += "\n";
  return crd + crds;
}

void Kubectl::ApplyObject(const json& meta, const json& object) {
  std::map<InstanceID, std::string> instances;
  for (auto const& kv : meta.items()) {
    InstanceID instance_id;
    std::stringstream(kv.key().substr(1)) >> instance_id;
    instances.emplace(instance_id,
                      kv.value()["nodename"].get_ref<std::string const&>());
  }

  std::string crds;
  if (object.value("global", false)) {
    crds = generate_global_object(instances, object);
  } else {
    crds = generate_local_object(instances, object);
  }
  VLOG(10) << "Apply CRDs: " << crds;
  this->Apply(crds, [](const Status& status) { return status; });
}

void Kubectl::Finish() {
  proc_->Finish();
  for (auto const& line : Diagnostic()) {
    VLOG(10) << "kubectl appy: " << line;
  }
  proc_->Wait();
  VLOG(10) << "kubectl exit with: " << proc_->ExitCode();
}

}  // namespace vineyard
