package com.text2text.digitizer

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.text2text.digitizer.model.SaleItem

class SaleItemAdapter(private val items: List<SaleItem>) :
    RecyclerView.Adapter<SaleItemAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvDescription: TextView = view.findViewById(R.id.tvDescription)
        val tvQuantity: TextView = view.findViewById(R.id.tvQuantity)
        val tvUnitPrice: TextView = view.findViewById(R.id.tvUnitPrice)
        val tvLineTotal: TextView = view.findViewById(R.id.tvLineTotal)
        val tvNotes: TextView = view.findViewById(R.id.tvNotes)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_sale_row, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        val currency = item.currency ?: ""
        holder.tvDescription.text = item.description
        holder.tvQuantity.text = "Qty: ${item.quantity}"
        holder.tvUnitPrice.text = "Unit: $currency${item.unit_price}"
        holder.tvLineTotal.text = "Total: $currency${item.line_total}"
        if (!item.notes.isNullOrBlank()) {
            holder.tvNotes.visibility = View.VISIBLE
            holder.tvNotes.text = item.notes
        } else {
            holder.tvNotes.visibility = View.GONE
        }
    }

    override fun getItemCount() = items.size
}
