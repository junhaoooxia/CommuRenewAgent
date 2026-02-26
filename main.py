from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from commurenew_agent.app import generate_design_schemes, index_knowledge_base
from commurenew_agent.models import PerceptionInput


def _setup_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Configure root logger so logs from all commurenew_agent modules land in the same log file.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger("commurenew_agent.main")
    logger.info("Logger initialized. Log file: %s", log_file)
    return logger


def _to_serializable_generation(output) -> dict:
    return {
        "scheme_list": [
            scheme.__dict__ | {"node_scenes": [scene.__dict__ for scene in scheme.node_scenes]}
            for scheme in output.scheme_list
        ]
    }


def _save_output(retrieval: dict, output_dict: dict, logger: logging.Logger, result_timestamp: str) -> Path:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{result_timestamp}.json"
    payload = {"retrieval": retrieval, "generation": output_dict}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved final result to: %s", output_path)
    return output_path


if __name__ == "__main__":
    logger = _setup_logger()
    result_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Step 1/4: Discovering knowledge sources under ./knowledge by file extension")
    knowledge_dir = Path("knowledge")
    source_specs = []
    if knowledge_dir.exists():
        for path in sorted(knowledge_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".pdf":
                stem = path.stem.lower()
                node_type = "policy" if "polic" in stem else ("design_method" if "design" in stem else ("trend_strategy" if "trend" in stem else "other"))
                source_specs.append({"source": "pdf", "pdf_path": str(path), "type": node_type})
            elif path.suffix.lower() == ".jsonl":
                default_type = "design_method" if "design_method" in path.stem else ("trend_strategy" if "trend_strategy" in path.stem else "other")
                source_specs.append({"source": "jsonl", "jsonl_path": str(path), "type": default_type})
            elif path.suffix.lower() in {".docx", ".doc"}:
                stem = path.stem.lower()
                node_type = "policy" if "polic" in stem else ("design_method" if "design" in stem else ("trend_strategy" if "trend" in stem else "other"))
                source_specs.append({"source": "word", "word_path": str(path), "type": node_type})

    detected_pdf_count = sum(1 for s in source_specs if s.get("source") == "pdf")
    detected_jsonl_count = sum(1 for s in source_specs if s.get("source") == "jsonl")
    detected_word_count = sum(1 for s in source_specs if s.get("source") == "word")
    logger.info(
        "Detected sources in knowledge/: pdf=%d, jsonl=%d, word=%d, total=%d",
        detected_pdf_count,
        detected_jsonl_count,
        detected_word_count,
        len(source_specs),
    )

    if source_specs:
        logger.info("Step 2/4: Running knowledge indexing")
        count = index_knowledge_base(source_specs)
        logger.info("Indexed %d nodes", count)
    else:
        logger.warning("No sources found under ./knowledge. Skipping indexing.")

    logger.info("Step 3/4: Building perception input and running retrieval + reasoning")
    perception = PerceptionInput(
        district_name="北京市海淀区学院路街道逸成东苑小区（东侧板楼区）",

        current_description=
        """
        逸成东苑位于北京市海淀区学院路街道，东临学清路、西临双清路，北接月泉路并与建清园南区相邻，
        南侧毗邻城华园、清枫华景园。小区处于北四环与北五环之间，属于海淀区学院路板块，位于中关村科学城
        核心辐射范围内，周边高校与科研机构密集（清华、北林、矿大等），社会结构整体呈现高教育、高知识密度
        的社区背景。中观交通条件优越，邻近昌平线学知园地铁站与学知园公交站，周边商业（新辰里购物中心等）
        与公园绿地资源（奥林匹克森林公园）完善。

        小区总用地约12.7公顷，整体容积率约2.38，绿化率约35%，2003年开盘、2006年前后完成交付，
        总户数约2800户。小区分为东西两区：东区为板楼区（本研究重点），绿化覆盖较高、公共空间资源相对丰富，
        由11栋板楼组成；西区为塔楼区，由9栋塔楼构成。整体属于典型封闭式住区形态，外围围墙/栅栏与内部道路系统
        共同界定边界，出入口设置门禁与保安岗亭，管理秩序总体稳定。

        东侧板楼区共设置5个主要出入口：北侧2处（其中一处临近塔楼区，另一处靠近1号楼），东侧3处
        （北侧两个可对外开放、南侧一处为仅车行出口）。入口空间形象朴素，统一设置小区标识与LOGO，
        识别度尚可，但功能复合性、空间引导性与公共活动承载能力有限，主要承担通行与管理职能。

        住区内部公共空间与绿化系统基础较好：绿化以宅前绿地、道路绿化与片状绿地为主，道路两侧多为行道树
        （落叶乔木）+灌木的配置，乔木体量大、树种较丰富，灌木修剪较整齐；内部设置3处水景与1处泳池空间，
        配合岩石、亭台等景观构筑物，形成一定景观层次。室外活动空间方面，小区内部有7处相对集中的活动场地：
        北侧双圆广场串联、轴线感强；水池景观及上游出水口植被茂密，并与儿童活动区、老年健身区复合；
        西侧/西南侧多处老年健身与开放广场（集体健身、跳操等）；中心为圆形活动广场，并配置环状挡雨廊，
        具备一定全天候使用潜力；东南侧设置小型开放广场与基础健身设施。总体“空间点位多、类型较全”，但设施更新程度不高。

        建筑与居住条件方面：住宅建筑外观偏老旧，外立面存在不同程度老化与破损，局部外墙被爬山虎覆盖；
        部分首层住户将空间改造为私人小院，形成半私密界面，但整洁度参差，局部存在积灰、闲置与杂物堆放。
        小区仍保留少量服务性底商（美容、房产中介等），具备一定生活服务，但规模小且分布零散。无障碍设施配置相对完善，
        每栋楼外均有带扶手的缓坡。楼内一层大堂空间尺度较开阔，但管理松散，自行车等物品停放随意，影响通行与空间品质。
        电梯运行总体平稳，但设备老化明显，缺乏红外感应等安全辅助系统，关门速度偏快，对老年人/儿童存在潜在安全隐患。
        市政供应（水、电、燃气）基本正常，虽完成暖气系统更换，但居民反映供暖效果仍不理想。

        交通组织方面，小区总体实现人车分流：机动车主要沿外圈道路行驶，内部以步行系统为主。
        住区设置4个地下车库出入口（东侧近出入口2处，南侧1处、北侧1处），地下停车位总体充足、可满足需求。
        地面机动车停车主要集中在东侧南北入口及南侧地下车库入口附近，组织相对明确。
        但道路设施存在老化：机动车道路局部破损开裂；步行道路铺装老旧（部分砖石铺装）且破损、显脏，步行环境品质有待提升。
        非机动车停放与充电是薄弱环节：仅在东侧南北入口附近有2处集中非机动车停车场，其余区域缺少统一规划停车棚，
        电动车/自行车常停在单元门口或公共通道附近，秩序杂乱；地面充电设施数量不足，仅2号楼西北侧有1处充电点，
        难以满足需求。

        环境管理与安全方面：垃圾分类回收点较多，并有专人管理，整体卫生状况较好，但局部绿地仍有零散垃圾。
        出入口门禁与保安亭齐全，监控覆盖率较高并配合巡逻，治安总体良好；外围围墙与栅栏形成封闭边界，安全性较强。
        """,

        problem_summary=
        """
        1）基础设施与舒适性短板
        - 步行铺装老旧、破损、显脏，局部路段影响通行体验与无障碍友好性；
        - 车行道路局部开裂破损；
        - 供暖效果不理想（虽已更换系统但舒适性不足）；
        - 电梯设备老化、缺少红外感应等安全辅助，关门速度快，对老幼不友好。

        2）非机动车系统缺失（运行痛点突出）
        - 集中停放点少、覆盖不足，缺少统一规划的车棚与导引；
        - 非机动车挤占单元门口/公共通道，秩序杂乱并可能影响消防疏散与通行；
        - 充电点极少（仅见2号楼西北侧1处），供需矛盾明显。

        3）公共服务与空间供给不均衡
        - 室外活动场地数量较多但设施更新不足、维护与适配性待提升；
        - 缺乏对居民开放的室内公共活动空间，公共交往高度依赖室外，雨雪/冬季使用受限。

        4）入口与共享空间品质不足
        - 出入口以通行管理为主，空间引导性、复合功能与社区界面表现弱；
        - 楼内大堂管理松散，随意停放影响秩序与空间品质；
        - 首层私家庭院化带来界面不统一，局部杂物堆放影响整体环境形象。

        5）智慧化设施缺口与治理升级需求
        - 智慧管理设施整体欠缺（智慧安防、智慧物业、智慧物流等），居民感知弱；
        - 居民对智慧化持开放态度，但对安全、隐私与噪音高度敏感，需要“可控、可解释、可管理”的导入路径。
        """,

        survey_summary=
        """
        问卷有效样本：17份（线上业主群+线下随机），一定程度覆盖不同年龄与使用习惯群体。

        居民画像要点
        - 性别比例均衡（男8、女9）；
        - 年龄以20-40岁为主，同时包含40-60岁中老年群体；
        - 受教育水平高：研究生及以上占比过半；
        - 收入中等偏上：月收入集中在1-2万元，家庭年收入分布较分散但整体具备一定支付能力；
        - 居住状态：自购约六成、租住约四成；居住年限10年以上占三成多，也有一定比例新住户。

        满意度总体判断
        - 整体处于“基本满意/一般”之间，但存在结构性短板；
        - 室内住房空间满意度最高；
        - 绿化景观满意度较高（原有35%绿化率、水景配置仍被认可）；
        - 地下停车位满意度较高（数量充足）。

        主要短板与痛点（问卷与现场一致）
        - 室内公共活动用房得分最低，且“不了解/无设施”比例高（反映缺失）；
        - 非机动车停放与充电满意度偏低（存在“不满意/非常不满意”与“无设施”反馈）；
        - 步行道路存在不满意比例，铺装老旧与破损影响体验；
        - 智慧管理设施评分偏低，居民普遍认为欠缺。

        更新迫切度（多选题）
        - 基础设施类（管网、路面铺装、无障碍等）最高：64.71%；
        - 智慧安防/智慧物流/智能管理：47.06%；
        - 建筑单体、公共空间、绿化景观均为：41.18%；
        - 交通停车整体焦虑不高（地下车位充足），但仍希望优化非机动车与充电：29.41%。

        未来场景接受度（低空经济/智慧物流）
        - 地面无人车/机器人配送接受度更高（>70%接受）；
        - 无人机配送接受度相对较高但分歧更大（>60%接受，近三成不太/不接受）；
        - 公共空间让渡意愿：近七成愿意用于相关设施建设（其余多为无所谓）。

            主要顾虑
            - 安全隐患最突出：76.47%（坠物/碰撞等）；
            - 隐私泄露：52.94%；
            - 噪音干扰：35.29%；
            - 对费用与占用公共空间的顾虑相对较低。

            设施选址偏好
            - 物流接收点更偏好“单元门口/大堂”：52.94%；
            - 其次为“中心花园/开敞空地”：23.53%；
            ——体现“就近、便捷”诉求，同时要求安全与隐私可控。
        """,

        site_images=[]
    )

    retrieval, output = generate_design_schemes(
        perception=perception,
        # Set to True after configuring GEMINI_API_KEY/GOOGLE_API_KEY for img2img outputs.
        generate_images=True,
        image_output_dir="output",
        result_timestamp=result_timestamp,
    )

    logger.info(
        "Retrieval done. methods=%d, policies=%d, strategies=%d",
        len(retrieval.get("retrieved_methods", [])),
        len(retrieval.get("retrieved_policies", [])),
        len(retrieval.get("retrieved_trend_strategies", [])),
    )
    logger.info("Reasoning done. generated_schemes=%d", len(output.scheme_list))

    logger.info("Step 4/4: Saving results to output directory")
    output_dict = _to_serializable_generation(output)
    saved_path = _save_output(retrieval, output_dict, logger, result_timestamp=result_timestamp)

    print("\n=== Retrieval ===")
    print(json.dumps(retrieval, ensure_ascii=False, indent=2))

    print("\n=== Generated Schemes ===")
    print(json.dumps(output_dict, ensure_ascii=False, indent=2))

    print(f"\nResult written to: {saved_path}")
