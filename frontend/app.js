document.addEventListener('DOMContentLoaded', async () => {
  // Navigation Tabs
  const navButtons = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('page-title');
  const pageSubtitle = document.getElementById('page-subtitle');

  const tabTitles = {
    'inspector-tab': {
      title: 'Pre-Dispatch Order Risk Inspector',
      subtitle: 'Real-time risk scoring, explainability reasons, and operational dispatch intervention'
    },
    'batch-tab': {
      title: 'Batch Risk Evaluator & Warehouse Queue',
      subtitle: 'Dispatch queue prioritization, risk tier segmentation, and bulk export'
    },
    'policy-tab': {
      title: 'Executive Policy & ROI Simulation Studio',
      subtitle: 'Dynamic cost-benefit modeling comparing Status Quo, Naive Holding, and Smart Calling'
    },
    'analytics-tab': {
      title: 'Model Performance & Validation Hub',
      subtitle: 'Cross-validation metrics, probability calibration, and feature weight attributions'
    }
  };

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      navButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(tabId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      if (tabTitles[tabId]) {
        pageTitle.textContent = tabTitles[tabId].title;
        pageSubtitle.textContent = tabTitles[tabId].subtitle;
      }
    });
  });

  // Theme Toggle
  const themeToggle = document.getElementById('theme-toggle');
  themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('theme-light');
  });

  // Load Presets
  let samplePresets = [];
  try {
    const res = await fetch('/api/sample_orders');
    if (res.ok) {
      samplePresets = await res.json();
      const presetSelect = document.getElementById('sample-preset-select');
      samplePresets.forEach((sample, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = sample.label;
        presetSelect.appendChild(opt);
      });

      presetSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val !== '') {
          populateFormWithSample(samplePresets[parseInt(val)].data);
          document.getElementById('btn-score-order').click();
        }
      });
    }
  } catch (err) {
    console.warn('Could not load sample presets:', err);
  }

  function populateFormWithSample(data) {
    document.getElementById('input-order-id').value = data.order_id || 'KO2612001';
    document.getElementById('input-order-placed-at').value = data.order_placed_at || '2026-10-01 14:30';
    document.getElementById('input-customer-id').value = data.customer_id || 'KC100983';
    document.getElementById('input-sku').value = data.sku || 'KH-RV-03';
    document.getElementById('input-sales-channel').value = data.sales_channel || 'marketplace';
    document.getElementById('input-payment-mode').value = data.payment_mode || 'cod';
    document.getElementById('input-discount-pct').value = data.discount_pct || 15;
    document.getElementById('input-qty').value = data.qty || 1;
    document.getElementById('input-order-value').value = data.order_value_inr || 24352.36;
    document.getElementById('input-promised-days').value = data.promised_delivery_days || 8;
    document.getElementById('input-pincode').value = data.delivery_pincode || 440314;
    document.getElementById('input-is-gift').value = data.is_gift || 'N';
    document.getElementById('input-prior-orders').value = data.customer_prior_orders || 3;
    document.getElementById('input-prior-returns').value = data.customer_prior_returns || 2;
    document.getElementById('input-delivery-note').value = data.delivery_note || '';
    document.getElementById('input-service-event').value = data.last_service_event_type || 'NONE';
    document.getElementById('input-source').value = data.source || 'crm';
  }

  // Single Order Form Submit
  const orderForm = document.getElementById('order-form');
  orderForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-score-order');
    btn.disabled = true;
    btn.innerHTML = `Evaluating...`;

    const payload = {
      order_id: document.getElementById('input-order-id').value,
      order_placed_at: document.getElementById('input-order-placed-at').value,
      customer_id: document.getElementById('input-customer-id').value,
      sku: document.getElementById('input-sku').value,
      sales_channel: document.getElementById('input-sales-channel').value,
      payment_mode: document.getElementById('input-payment-mode').value,
      discount_pct: parseFloat(document.getElementById('input-discount-pct').value),
      qty: parseInt(document.getElementById('input-qty').value),
      order_value_inr: parseFloat(document.getElementById('input-order-value').value),
      promised_delivery_days: parseInt(document.getElementById('input-promised-days').value),
      delivery_pincode: parseInt(document.getElementById('input-pincode').value),
      is_gift: document.getElementById('input-is-gift').value,
      customer_prior_orders: parseInt(document.getElementById('input-prior-orders').value),
      customer_prior_returns: parseInt(document.getElementById('input-prior-returns').value),
      delivery_note: document.getElementById('input-delivery-note').value,
      last_service_event_type: document.getElementById('input-service-event').value,
      source: document.getElementById('input-source').value
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const result = await res.json();
        updateResultsUI(result);
      } else {
        alert('Prediction request failed.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error while evaluating order.');
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Evaluate Return Risk`;
    }
  });

  function updateResultsUI(data) {
    const pct = data.return_probability_pct;
    const score = data.return_probability;

    // Badge
    const badge = document.getElementById('risk-tier-badge');
    badge.textContent = data.risk_badge;
    badge.className = 'badge';
    if (data.risk_tier === 'CRITICAL_RISK' || data.risk_tier === 'HIGH_RISK') {
      badge.classList.add('badge-danger');
    } else if (data.risk_tier === 'MODERATE_RISK') {
      badge.classList.add('badge-warning');
    } else {
      badge.classList.add('badge-success');
    }

    // Gauge
    const circle = document.getElementById('gauge-circle');
    circle.setAttribute('stroke-dasharray', `${pct}, 100`);
    if (score >= 0.28) {
      circle.style.stroke = 'var(--color-danger)';
    } else if (score >= 0.115) {
      circle.style.stroke = 'var(--color-warning)';
    } else {
      circle.style.stroke = 'var(--color-success)';
    }

    document.getElementById('score-percentage').textContent = `${pct}%`;
    document.getElementById('raw-score-text').textContent = `Score: ${score.toFixed(4)}`;

    // Action Banner
    const banner = document.getElementById('action-banner');
    banner.className = 'action-banner';
    if (data.risk_tier === 'CRITICAL_RISK' || data.risk_tier === 'HIGH_RISK') {
      banner.classList.add('action-danger');
    } else if (data.risk_tier === 'MODERATE_RISK') {
      banner.classList.add('action-warning');
    } else {
      banner.classList.add('action-success');
    }

    document.getElementById('action-title').textContent = data.recommended_action;
    document.getElementById('action-desc').textContent = data.action_description;

    // Reasons List
    const reasonsList = document.getElementById('reasons-list');
    reasonsList.innerHTML = '';
    if (data.human_readable_reasons && data.human_readable_reasons.length > 0) {
      data.human_readable_reasons.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'Standard low-risk profile: zero prior returns and fast delivery window.';
      li.style.borderLeftColor = 'var(--color-success)';
      reasonsList.appendChild(li);
    }

    // Economics
    document.getElementById('econ-loss').textContent = `₹${data.economics.expected_return_cost_inr.toFixed(2)}`;
    document.getElementById('econ-cost').textContent = `₹${data.economics.call_cost_inr.toFixed(2)}`;
    document.getElementById('econ-savings').textContent = `+₹${data.economics.net_savings_if_called_inr.toFixed(2)}`;
    document.getElementById('econ-roi').textContent = `+${data.economics.roi_on_confirmation_call.toFixed(1)}%`;
  }

  // Load Model Info & Analytics
  try {
    const res = await fetch('/api/model_info');
    if (res.ok) {
      const info = await res.json();
      
      // Update Analytics Tab KPIs
      if (info.metrics) {
        document.getElementById('metric-auc').textContent = info.metrics.roc_auc.toFixed(4);
        document.getElementById('metric-prauc').textContent = info.metrics.pr_auc.toFixed(4);
        document.getElementById('metric-brier').textContent = info.metrics.brier_score.toFixed(4);
        document.getElementById('metric-train-orders').textContent = info.metrics.n_train_orders.toLocaleString();
      }

      // Feature Weights
      const featContainer = document.getElementById('feature-weights-container');
      if (featContainer && info.top_feature_coefficients) {
        featContainer.innerHTML = '';
        info.top_feature_coefficients.forEach(item => {
          const div = document.createElement('div');
          div.className = 'feature-bar-item';
          const isPos = item.coefficient > 0;
          div.innerHTML = `
            <span class="name">${item.feature}</span>
            <span class="weight ${isPos ? 'text-danger' : 'text-success'}">${isPos ? '+' : ''}${item.coefficient.toFixed(4)}</span>
          `;
          featContainer.appendChild(div);
        });
      }

      // Calibration Table
      const calibTbody = document.getElementById('calibration-tbody');
      if (calibTbody && info.evaluation_report && info.evaluation_report.calibration_curve) {
        calibTbody.innerHTML = '';
        info.evaluation_report.calibration_curve.forEach(bin => {
          const tr = document.createElement('tr');
          const predPct = (bin.mean_predicted_risk * 100).toFixed(1);
          const truePct = (bin.actual_return_rate * 100).toFixed(1);
          const diff = Math.abs(bin.mean_predicted_risk - bin.actual_return_rate);
          const matchBadge = diff < 0.05 ? `<span class="badge badge-success">Accurate</span>` : `<span class="badge badge-warning">Fair</span>`;
          
          tr.innerHTML = `
            <td><strong>${bin.bin_range}</strong></td>
            <td>${bin.order_count.toLocaleString()}</td>
            <td>${predPct}%</td>
            <td><strong>${truePct}%</strong></td>
            <td>${matchBadge}</td>
          `;
          calibTbody.appendChild(tr);
        });
      }
    }
  } catch (err) {
    console.warn('Could not fetch model info:', err);
  }

  // Load Batch Sample Data
  loadBatchSnapshot();

  async function loadBatchSnapshot() {
    const tableBody = document.getElementById('batch-table-body');
    try {
      // Fetch sample presets and populate initial table rows
      if (samplePresets.length > 0) {
        tableBody.innerHTML = '';
        for (const sample of samplePresets) {
          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sample.data)
          });
          if (res.ok) {
            const pred = await res.json();
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${pred.order_id}</strong></td>
              <td>${sample.data.customer_id}</td>
              <td>${sample.data.sku}</td>
              <td>${sample.data.sales_channel}</td>
              <td>${sample.data.payment_mode.toUpperCase()}</td>
              <td>₹${sample.data.order_value_inr.toLocaleString()}</td>
              <td><strong>${pred.return_probability_pct}%</strong></td>
              <td><span class="badge ${pred.risk_tier.includes('HIGH') || pred.risk_tier.includes('CRITICAL') ? 'badge-danger' : (pred.risk_tier.includes('MODERATE') ? 'badge-warning' : 'badge-success')}">${pred.risk_tier}</span></td>
              <td>${pred.recommended_action}</td>
            `;
            tableBody.appendChild(tr);
          }
        }
      }
    } catch (err) {
      console.warn('Error populating batch snapshot table:', err);
    }
  }

  // Policy Simulator Sliders
  const callSlider = document.getElementById('slider-call-thresh');
  const holdSlider = document.getElementById('slider-hold-thresh');
  const valCall = document.getElementById('val-call-thresh');
  const valHold = document.getElementById('val-hold-thresh');

  callSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valCall.textContent = `${val.toFixed(3)} (${(val * 100).toFixed(1)}%)`;
  });

  holdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valHold.textContent = `${val.toFixed(3)} (${(val * 100).toFixed(1)}%)`;
  });

  // Download Predictions Button
  document.getElementById('btn-download-preds').addEventListener('click', () => {
    window.location.href = '/static/../predictions.csv';
  });

  // Trigger initial score calculation
  if (document.getElementById('btn-score-order')) {
    document.getElementById('btn-score-order').click();
  }
});
